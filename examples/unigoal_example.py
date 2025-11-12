"""
Example usage of DualMap-UniGoal integration

This script demonstrates how to use the DualMap-UniGoal bridge for
semantic navigation tasks.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dualmap import Dualmap
from utils.unigoal_bridge import DualMapUniGoalBridge, SemanticObject
from omegaconf import OmegaConf


def example_basic_integration():
    """Example 1: Basic integration setup"""
    print("=" * 60)
    print("Example 1: Basic DualMap-UniGoal Integration")
    print("=" * 60)

    # Load configuration
    cfg = OmegaConf.load("config/runner_unigoal.yaml")

    # Initialize DualMap
    print("\n[1] Initializing DualMap...")
    dualmap = Dualmap(cfg)
    print("✓ DualMap initialized")

    # Initialize bridge
    print("\n[2] Initializing UniGoal bridge...")
    unigoal_config = OmegaConf.to_container(cfg.unigoal, resolve=True)
    bridge = DualMapUniGoalBridge(dualmap, unigoal_config)
    print("✓ Bridge initialized")

    # Set a navigation goal
    print("\n[3] Setting navigation goal...")
    goal = "找到厨房里的咖啡机"  # "Find the coffee machine in the kitchen"
    success = bridge.set_navigation_goal(goal)

    if success:
        print(f"✓ Goal set: '{goal}'")
        status = bridge.get_status()
        print(f"  - Sub-goals: {status['total_sub_goals']}")
        print(f"  - Current sub-goal: {status['current_sub_goal']}")
    else:
        print("✗ Failed to set goal")

    print("\n" + "=" * 60)


def example_object_query():
    """Example 2: Query objects using natural language"""
    print("=" * 60)
    print("Example 2: Natural Language Object Query")
    print("=" * 60)

    # Load configuration
    cfg = OmegaConf.load("config/runner_unigoal.yaml")

    # Initialize DualMap with some mock objects
    dualmap = Dualmap(cfg)

    # Create bridge
    bridge = DualMapUniGoalBridge(dualmap, {})

    # Mock some semantic objects for demonstration
    print("\n[1] Creating mock semantic objects...")
    mock_objects = [
        SemanticObject(
            uid="obj_001",
            class_name="coffee machine",
            position=np.array([1.5, 2.0, 0.8]),
            bbox=np.array([1.4, 1.9, 0.7, 1.6, 2.1, 0.9]),
            clip_feature=np.random.randn(512),  # Mock CLIP feature
        ),
        SemanticObject(
            uid="obj_002",
            class_name="refrigerator",
            position=np.array([0.5, 3.0, 0.0]),
            bbox=np.array([0.3, 2.8, 0.0, 0.7, 3.2, 1.8]),
            clip_feature=np.random.randn(512),
        ),
        SemanticObject(
            uid="obj_003",
            class_name="dining table",
            position=np.array([3.0, 2.5, 0.0]),
            bbox=np.array([2.5, 2.0, 0.0, 3.5, 3.0, 0.8]),
            clip_feature=np.random.randn(512),
        ),
    ]

    print(f"✓ Created {len(mock_objects)} mock objects")

    # Query objects
    print("\n[2] Querying objects with natural language...")
    queries = [
        "咖啡机",  # coffee machine
        "冰箱",    # refrigerator
        "桌子",    # table
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        # Note: In real usage, this would use actual CLIP features
        # For now, we just demonstrate the structure
        print(f"  → This would match objects based on CLIP similarity")

    print("\n" + "=" * 60)


def example_navigation_workflow():
    """Example 3: Complete navigation workflow"""
    print("=" * 60)
    print("Example 3: Complete Navigation Workflow")
    print("=" * 60)

    # Load configuration
    cfg = OmegaConf.load("config/runner_unigoal.yaml")

    print("\n[1] Initializing system...")
    dualmap = Dualmap(cfg)
    unigoal_config = OmegaConf.to_container(cfg.unigoal, resolve=True)
    bridge = DualMapUniGoalBridge(dualmap, unigoal_config)
    print("✓ System initialized")

    # Set complex goal
    print("\n[2] Setting complex navigation goal...")
    goal = "先去厨房找到咖啡机,然后去客厅找到沙发"
    # "First go to the kitchen to find the coffee machine, then go to the living room to find the sofa"

    success = bridge.set_navigation_goal(goal)

    if success:
        print(f"✓ Complex goal set: '{goal}'")

        # Show decomposition
        status = bridge.get_status()
        print(f"\n[3] Goal decomposition:")
        print(f"  - Total sub-goals: {status['total_sub_goals']}")

        # Simulate navigation steps
        print(f"\n[4] Simulating navigation...")
        current_pos = np.array([0.0, 0.0, 0.0])

        for step in range(5):  # Simulate 5 steps
            action = bridge.execute_navigation(current_pos)

            print(f"\n  Step {step + 1}:")
            print(f"    - Current sub-goal: {status['current_sub_goal']}")
            print(f"    - Action: {action}")

            if action.get('done', False):
                print(f"    ✓ Navigation completed!")
                break

            # Update position (mock movement)
            if 'linear' in action:
                current_pos[0] += action['linear'] * 0.1

    print("\n" + "=" * 60)


def example_config_options():
    """Example 4: Different configuration options"""
    print("=" * 60)
    print("Example 4: Configuration Options")
    print("=" * 60)

    print("\n[1] LLM Backend Options:")
    print("  - Ollama (local): Good for privacy, requires local setup")
    print("  - OpenAI: More powerful, requires API key")
    print("  - None: Simple rule-based decomposition")

    print("\n[2] Example Ollama configuration:")
    print("""
    unigoal:
      use_llm_decomposition: True
      llm_backend: "ollama"
      llm_model: "llama3"
      ollama:
        host: "http://localhost:11434"
    """)

    print("\n[3] Example OpenAI configuration:")
    print("""
    unigoal:
      use_llm_decomposition: True
      llm_backend: "openai"
      openai:
        api_key: "your-api-key"
        model: "gpt-4"
    """)

    print("\n[4] Planning method options:")
    print("  - fmm: Fast Marching Method (good for grid-based planning)")
    print("  - astar: A* algorithm (classic pathfinding)")

    print("\n[5] Goal modes:")
    print("  - inquiry: Simple text-based goal")
    print("  - llm_inquiry: LLM-enhanced goal decomposition")
    print("  - image_goal: Use reference image as target")
    print("  - random: Random exploration")

    print("\n" + "=" * 60)


def example_integration_benefits():
    """Example 5: Benefits of integration"""
    print("=" * 60)
    print("Example 5: Benefits of DualMap-UniGoal Integration")
    print("=" * 60)

    print("\n[1] DualMap Contributions:")
    print("  ✓ Real-time semantic mapping")
    print("  ✓ Open-vocabulary object detection (YOLO-World)")
    print("  ✓ Dual-level map representation (local + global)")
    print("  ✓ CLIP-based semantic understanding")
    print("  ✓ Dynamic scene handling")

    print("\n[2] UniGoal Contributions:")
    print("  ✓ Zero-shot navigation (no task-specific training)")
    print("  ✓ LLM-based goal decomposition")
    print("  ✓ Multi-modal goal specification")
    print("  ✓ Graph-based reasoning")
    print("  ✓ Robust path planning (FMM)")

    print("\n[3] Integration Benefits:")
    print("  ✓ Semantic understanding + Intelligent navigation")
    print("  ✓ Natural language task specification")
    print("  ✓ Complex goal decomposition and execution")
    print("  ✓ Real-time mapping with zero-shot decision making")
    print("  ✓ Support for both simulation and real robots")

    print("\n[4] Example Use Cases:")
    print("  - Home service robots: 'Clean up the kitchen then water the plants'")
    print("  - Warehouse robots: 'Find the red box on shelf B and bring it to station 3'")
    print("  - Exploration robots: 'Survey the area and identify all furniture'")

    print("\n" + "=" * 60)


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DualMap-UniGoal Integration Examples")
    print("=" * 60)

    examples = [
        ("Basic Integration", example_basic_integration),
        ("Object Query", example_object_query),
        ("Navigation Workflow", example_navigation_workflow),
        ("Configuration Options", example_config_options),
        ("Integration Benefits", example_integration_benefits),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"\n✗ Error in '{name}': {e}\n")

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
