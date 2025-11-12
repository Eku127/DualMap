"""
DualMap-UniGoal Integration Runner

This runner integrates DualMap's semantic mapping with UniGoal's zero-shot
navigation capabilities for embodied AI navigation tasks.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
from pathlib import Path
import time
from typing import Optional, Dict, Any

# DualMap imports
from dualmap import Dualmap
from utils.unigoal_bridge import DualMapUniGoalBridge
from utils.types import DataInput
from utils.data_loader import dataset_initialization
from utils.visualizer import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UniGoalRunner:
    """Runner for DualMap-UniGoal integrated navigation"""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.dualmap = None
        self.bridge = None
        self.visualizer = None

        # Navigation state
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_heading = 0.0
        self.navigation_active = False

        # Statistics
        self.stats = {
            'total_steps': 0,
            'successful_goals': 0,
            'failed_goals': 0,
            'total_distance': 0.0,
            'start_time': None,
        }

    def initialize(self):
        """Initialize DualMap and UniGoal bridge"""
        logger.info("Initializing DualMap-UniGoal system...")

        # Initialize DualMap
        self.dualmap = Dualmap(self.cfg)
        logger.info("DualMap initialized")

        # Initialize bridge
        unigoal_config = OmegaConf.to_container(self.cfg.unigoal, resolve=True)
        self.bridge = DualMapUniGoalBridge(self.dualmap, unigoal_config)
        logger.info("UniGoal bridge initialized")

        # Initialize visualizer if enabled
        if self.cfg.dualmap.use_rerun:
            self.visualizer = Visualizer.get_instance()
            logger.info("Rerun visualizer initialized")

        self.stats['start_time'] = time.time()

    def process_frame(self, data_input: DataInput) -> bool:
        """
        Process a single frame through DualMap

        Args:
            data_input: Input data containing RGB, depth, pose, etc.

        Returns:
            True if processing was successful
        """
        try:
            # Update current position from pose
            if data_input.pose is not None:
                self.current_position = data_input.pose[:3, 3]

            # Process through DualMap
            if self.cfg.dualmap.use_parallel:
                self.dualmap.parallel_process(data_input)
            else:
                self.dualmap.sequential_process(data_input)

            self.stats['total_steps'] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False

    def set_navigation_goal(self, goal: str) -> bool:
        """
        Set a navigation goal

        Args:
            goal: Natural language goal description

        Returns:
            True if goal was set successfully
        """
        logger.info(f"Setting navigation goal: '{goal}'")

        success = self.bridge.set_navigation_goal(goal)

        if success:
            self.navigation_active = True
            logger.info("Navigation goal set successfully")
        else:
            logger.error("Failed to set navigation goal")

        return success

    def execute_navigation_step(self) -> Dict[str, Any]:
        """
        Execute one step of navigation

        Returns:
            Action dictionary with navigation commands
        """
        if not self.navigation_active:
            return {'done': True, 'message': 'No active navigation'}

        # Execute navigation through bridge
        action = self.bridge.execute_navigation(
            self.current_position,
            self.current_heading
        )

        # Check if navigation completed
        if action.get('done', False):
            self.navigation_active = False

            if 'error' in action:
                self.stats['failed_goals'] += 1
                logger.warning(f"Navigation failed: {action.get('message', 'Unknown error')}")
            else:
                self.stats['successful_goals'] += 1
                logger.info(f"Navigation succeeded: {action.get('message', 'Goal reached')}")

        # Update statistics
        if 'linear' in action:
            self.stats['total_distance'] += abs(action['linear']) * 0.1  # Assuming 10Hz

        return action

    def run_dataset_mode(self):
        """Run with offline dataset"""
        logger.info("Running in dataset mode...")

        # Initialize dataset
        dataset_loader = dataset_initialization(self.cfg)

        if dataset_loader is None:
            logger.error("Failed to initialize dataset")
            return

        total_frames = len(dataset_loader)
        logger.info(f"Processing {total_frames} frames from dataset")

        # Process all frames first to build map
        logger.info("Building semantic map...")
        for idx, data_input in enumerate(dataset_loader):
            if idx % 10 == 0:
                logger.info(f"Processing frame {idx}/{total_frames}")

            success = self.process_frame(data_input)

            if not success:
                logger.warning(f"Failed to process frame {idx}")

        # Finalize map
        self.dualmap.end_process()
        logger.info("Map building completed")

        # Now run navigation task if goal is specified
        if self.cfg.navigation.goal_mode != "none":
            self.run_navigation_task()

    def run_navigation_task(self):
        """Run navigation task with current map"""
        logger.info("Starting navigation task...")

        # Get goal based on goal mode
        goal = self.get_navigation_goal()

        if goal is None:
            logger.warning("No navigation goal specified")
            return

        # Set navigation goal
        self.set_navigation_goal(goal)

        # Execute navigation
        max_steps = self.cfg.navigation.max_steps
        for step in range(max_steps):
            action = self.execute_navigation_step()

            # Log progress
            if step % 10 == 0:
                status = self.bridge.get_status()
                logger.info(f"Step {step}: {status}")

            # Check if done
            if action.get('done', False):
                logger.info(f"Navigation completed at step {step}")
                break

            # Visualize if enabled
            if self.visualizer is not None:
                self.visualize_navigation_state(action)

            time.sleep(0.1)  # Simulate real-time execution

        # Print final statistics
        self.print_statistics()

    def get_navigation_goal(self) -> Optional[str]:
        """
        Get navigation goal based on goal mode

        Returns:
            Goal description or None
        """
        goal_mode = self.cfg.navigation.goal_mode

        if goal_mode == "inquiry" or goal_mode == "llm_inquiry":
            # Read from config or prompt user
            goal = self.cfg.get('inquiry_sentence', None)

            if goal is None:
                logger.info("Enter navigation goal:")
                goal = input("> ")

            return goal

        elif goal_mode == "random":
            # Select random object from map
            objects = self.bridge.dualmap_interface.get_global_objects()

            if not objects:
                logger.warning("No objects in map for random goal")
                return None

            import random
            target_obj = random.choice(objects)
            return f"navigate to {target_obj.class_name}"

        else:
            logger.warning(f"Unsupported goal mode: {goal_mode}")
            return None

    def visualize_navigation_state(self, action: Dict[str, Any]):
        """Visualize current navigation state"""
        if self.visualizer is None:
            return

        # Visualize planned path
        if self.bridge.current_path is not None:
            path_3d = np.column_stack([
                self.bridge.current_path,
                np.zeros(len(self.bridge.current_path))  # z=0
            ])
            # Log path to Rerun (visualization code here)

        # Visualize current position and heading
        # (visualization code here)

    def print_statistics(self):
        """Print navigation statistics"""
        elapsed_time = time.time() - self.stats['start_time']

        logger.info("=" * 50)
        logger.info("Navigation Statistics:")
        logger.info(f"  Total steps: {self.stats['total_steps']}")
        logger.info(f"  Successful goals: {self.stats['successful_goals']}")
        logger.info(f"  Failed goals: {self.stats['failed_goals']}")
        logger.info(f"  Total distance: {self.stats['total_distance']:.2f} m")
        logger.info(f"  Elapsed time: {elapsed_time:.2f} s")

        if self.stats['successful_goals'] > 0:
            success_rate = self.stats['successful_goals'] / (
                self.stats['successful_goals'] + self.stats['failed_goals']
            )
            logger.info(f"  Success rate: {success_rate * 100:.1f}%")

        logger.info("=" * 50)

    def run_ros_mode(self):
        """Run with ROS (to be implemented)"""
        logger.warning("ROS mode not yet implemented in UniGoal runner")
        logger.info("Please use the standard runner_ros.py for ROS integration")

    def run_habitat_mode(self):
        """Run with Habitat simulator (to be implemented)"""
        logger.warning("Habitat mode not yet implemented")
        logger.info("This mode will integrate with Habitat Data Collector")

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")

        if self.dualmap is not None:
            self.dualmap.end_process()

        logger.info("Cleanup completed")


@hydra.main(version_base=None, config_path="../config", config_name="runner_unigoal")
def main(cfg: DictConfig):
    """Main entry point"""
    logger.info("DualMap-UniGoal Integration Runner")
    logger.info("=" * 50)

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Create runner
    runner = UniGoalRunner(cfg)

    try:
        # Initialize
        runner.initialize()

        # Run based on mode
        if cfg.dataset.get('enable', True):
            runner.run_dataset_mode()
        elif cfg.ros.get('enable', False):
            runner.run_ros_mode()
        elif cfg.habitat.get('enable', False):
            runner.run_habitat_mode()
        else:
            logger.error("No running mode enabled. Please enable dataset, ros, or habitat mode.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)

    finally:
        # Cleanup
        runner.cleanup()

    logger.info("Runner finished")


if __name__ == "__main__":
    main()
