#!/bin/bash

# Set up signal handling
trap 'cleanup' INT TERM

# Global variable to store child PID
CHILD_PID=""

cleanup() {
    echo -e "\nReceived termination signal. Cleaning up..."
    if [ ! -z "$CHILD_PID" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null
        wait "$CHILD_PID" 2>/dev/null
    fi
    exit 1
}

# Create the initial HTML file
cat << 'EOF' > monitor.html
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Processing Metrics - ScanNet</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <meta http-equiv="refresh" content="5">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .plot-container { 
            width: 100%;
            max-width: 1200px;
            margin: 20px auto;
            height: 400px;
        }
        .current-stats {
            margin: 20px auto;
            max-width: 1200px;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-weight: bold;
            color: #2c3e50;
        }
        .sequence-name {
            color: #e74c3c;
            font-size: 1.5em;
        }
        .stats-table-container {
            margin: 20px auto;
            max-width: 1200px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-avg {
            color: #2980b9;
        }
        .metric-peak {
            color: #c0392b;
        }
    </style>
</head>
<body>
    <div class="current-stats" id="currentStats"></div>
    <div class="stats-table-container">
        <h2>Completed Sequences Statistics</h2>
        <table id="statsTable">
            <thead>
                <tr>
                    <th>Sequence</th>
                    <th>Processing Time (s)</th>
                    <th>Memory Usage (MB)</th>
                    <th>CPU Usage (%)</th>
                    <th>GPU Memory (MB)</th>
                    <th>GPU Utilization (%)</th>
                </tr>
            </thead>
            <tbody id="statsTableBody"></tbody>
        </table>
    </div>
    <div id="memPlot" class="plot-container"></div>
    <div id="cpuPlot" class="plot-container"></div>
    <div id="gpuMemPlot" class="plot-container"></div>
    <div id="gpuUtilPlot" class="plot-container"></div>

    <script>
        function calculateStats(data, key) {
            if (!data || !data.length) return { avg: 0, peak: 0 };
            const sum = data.reduce((a, b) => a + b, 0);
            const avg = sum / data.length;
            const peak = Math.max(...data);
            return { avg: avg.toFixed(2), peak: peak.toFixed(2) };
        }

        function updateStatsTable(data) {
            const tableBody = document.getElementById('statsTableBody');
            const currentSeq = data.currentSequence;
            let html = '';

            for (const seq in data) {
                if (seq === 'currentSequence') continue;

                const stats = data[seq];
                if (!stats || !stats.time.length) continue;

                const processingTime = stats.time[stats.time.length - 1].toFixed(2);
                const memStats = calculateStats(stats.memory);
                const cpuStats = calculateStats(stats.cpu);
                const gpuMemStats = calculateStats(stats.gpuMem);
                const gpuUtilStats = calculateStats(stats.gpuUtil);

                html += `
                    <tr>
                        <td>${seq}</td>
                        <td>${processingTime}</td>
                        <td>
                            <span class="metric-avg">Avg: ${memStats.avg}</span><br>
                            <span class="metric-peak">Peak: ${memStats.peak}</span>
                        </td>
                        <td>
                            <span class="metric-avg">Avg: ${cpuStats.avg}</span><br>
                            <span class="metric-peak">Peak: ${cpuStats.peak}</span>
                        </td>
                        <td>
                            <span class="metric-avg">Avg: ${gpuMemStats.avg}</span><br>
                            <span class="metric-peak">Peak: ${gpuMemStats.peak}</span>
                        </td>
                        <td>
                            <span class="metric-avg">Avg: ${gpuUtilStats.avg}</span><br>
                            <span class="metric-peak">Peak: ${gpuUtilStats.peak}</span>
                        </td>
                    </tr>
                `;
            }
            tableBody.innerHTML = html;
        }

        function loadData() {
            fetch('metrics_current.json')
                .then(response => response.json())
                .then(data => {
                    createPlots(data);
                    updateCurrentStats(data);
                    updateStatsTable(data);
                })
                .catch(error => console.error('Error loading data:', error));
        }

        function updateCurrentStats(data) {
            const currentSeq = data.currentSequence;
            const stats = data[currentSeq];
            const currentStatsDiv = document.getElementById('currentStats');
            
            if (!stats || currentSeq === "completed") {
                currentStatsDiv.innerHTML = `
                    <h2>Processing Complete</h2>
                    <p>All sequences have been processed.</p>
                `;
                return;
            }

            const lastIndex = stats.time.length - 1;
            const html = `
                <h2>Currently Processing: <span class="sequence-name">${currentSeq}</span></h2>
                <p>Memory Usage: <span class="stat-value">${stats.memory[lastIndex]} MB</span></p>
                <p>CPU Usage: <span class="stat-value">${stats.cpu[lastIndex]}%</span></p>
                <p>GPU Memory: <span class="stat-value">${stats.gpuMem[lastIndex]} MB</span></p>
                <p>GPU Utilization: <span class="stat-value">${stats.gpuUtil[lastIndex]}%</span></p>
                <p>Elapsed Time: <span class="stat-value">${stats.time[lastIndex]} seconds</span></p>
            `;
            currentStatsDiv.innerHTML = html;
        }

        function createPlots(data) {
            const plots = {
                'memPlot': {title: 'Memory Usage Over Time', yaxis: 'Memory (MB)', key: 'memory'},
                'cpuPlot': {title: 'CPU Usage Over Time', yaxis: 'CPU Usage (%)', key: 'cpu'},
                'gpuMemPlot': {title: 'GPU Memory Usage Over Time', yaxis: 'GPU Memory (MB)', key: 'gpuMem'},
                'gpuUtilPlot': {title: 'GPU Utilization Over Time', yaxis: 'GPU Utilization (%)', key: 'gpuUtil'}
            };

            for (const [elementId, config] of Object.entries(plots)) {
                const traces = [];
                
                for (const seq in data) {
                    if (seq === 'currentSequence') continue;
                    traces.push({
                        name: seq,
                        x: data[seq].time,
                        y: data[seq][config.key],
                        type: 'scatter',
                        mode: 'lines'
                    });
                }

                const layout = {
                    title: config.title,
                    xaxis: {title: 'Time (s)'},
                    yaxis: {title: config.yaxis},
                    showlegend: true,
                    template: 'plotly_white'
                };

                Plotly.newPlot(elementId, traces, layout);
            }
        }

        // Initial load and setup periodic refresh
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>
EOF

sequences=(
    "office_0" 
    "office_1" 
    "office_2" 
    "office_3" 
    "office_4" 
    "room_0" 
    "room_1" 
    "room_2"
)

THRESHOLD="1.2"


# Function to get GPU metrics
get_gpu_metrics() {
    local cuda_gpu="${CUDA_VISIBLE_DEVICES:-0}"
    cuda_gpu=$(echo "$cuda_gpu" | cut -d',' -f1)
    
    if [ -z "$cuda_gpu" ] || [ "$cuda_gpu" = "-1" ]; then
        echo "0,0"
        return
    fi

    local metrics=$(nvidia-smi --id="$cuda_gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    if [ ! -z "$metrics" ]; then
        echo "$metrics" | sed 's/, */,/g'
    else
        echo "0,0"
    fi
}

# Function to get CPU usage
get_cpu_usage() {
    local usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
    if [ ! -z "$usage" ]; then
        echo "$usage"
    else
        echo "0"
    fi
}

# Function to get memory usage
get_memory_usage() {
    local mem=$(free -m | awk 'NR==2{print $3}')
    if [ ! -z "$mem" ]; then
        echo "$mem"
    else
        echo "0"
    fi
}

# Function to merge metrics
merge_metrics_json() {
    if [ -f metrics_current.json ]; then
        mv metrics_current.json metrics_previous.json
        jq -s '.[0] * .[1]' metrics_previous.json metrics_new.json > metrics_current.json
        rm metrics_previous.json metrics_new.json
    else
        mv metrics_new.json metrics_current.json
    fi
}

# Function to update metrics JSON file
update_metrics_json() {
    local seq=$1
    local start_time=$2
    local metrics_dir=$3
    
    for file in "memory.txt" "cpu.txt" "gpu.txt"; do
        if [ ! -s "${metrics_dir}/${file}" ]; then
            echo "{\"currentSequence\": \"$seq\", \"$seq\": {\"time\": [], \"memory\": [], \"cpu\": [], \"gpuMem\": [], \"gpuUtil\": []}}" > metrics_new.json
            echo "Warning: ${file} is empty, creating empty JSON structure"
            merge_metrics_json
            return
        fi
    done
    
    awk '{printf "[%d, %d]\n", $1, $2}' "${metrics_dir}/memory.txt" > "${metrics_dir}/memory_formatted.txt"
    awk '{printf "[%d, %f]\n", $1, $2}' "${metrics_dir}/cpu.txt" > "${metrics_dir}/cpu_formatted.txt"
    awk '{
        split($2, values, ",")
        mem = values[1]
        util = values[2]
        if (util == "") util = 0
        printf "[%d, %f, %f]\n", $1, mem, util
    }' "${metrics_dir}/gpu.txt" > "${metrics_dir}/gpu_formatted.txt"
    
    jq -n \
        --arg seq "$seq" \
        --arg start "$start_time" \
        --slurpfile memory "${metrics_dir}/memory_formatted.txt" \
        --slurpfile cpu "${metrics_dir}/cpu_formatted.txt" \
        --slurpfile gpu "${metrics_dir}/gpu_formatted.txt" \
        '
        {
            currentSequence: $seq,
            ($seq): {
                time: [($memory // [])[] | .[0] - ($start | tonumber)],
                memory: [($memory // [])[] | .[1]],
                cpu: [($cpu // [])[] | .[1]],
                gpuMem: [($gpu // [])[] | .[1]],
                gpuUtil: [($gpu // [])[] | .[2]]
            }
        }
        ' > metrics_new.json 2>"${metrics_dir}/jq_error.log"
    
    if [ -s metrics_new.json ]; then
        merge_metrics_json
    else
        echo "{\"currentSequence\": \"$seq\", \"$seq\": {\"time\": [], \"memory\": [], \"cpu\": [], \"gpuMem\": [], \"gpuUtil\": []}}" > metrics_new.json
        echo "Warning: Failed to create JSON, using empty structure"
        merge_metrics_json
    fi
    
    rm -f "${metrics_dir}/memory_formatted.txt" "${metrics_dir}/cpu_formatted.txt" "${metrics_dir}/gpu_formatted.txt"
}

# Create directory for all metrics
mkdir -p metrics_data

# Process each sequence
for seq in "${sequences[@]}"; do
    echo "Processing sequence: $seq"
    
    metrics_dir="metrics_data_replica_mobileclip/${seq}"
    mkdir -p "$metrics_dir"
    
    start_time=$(date +%s)
    
    # # Start the main process with ScanNet parameters
    # python application/semantic_segmentation.py main.dataset=scannet "main.dataset_path=scannet/${seq}/" "main.save_path=data/sem_seg_new_scannet_mobileclip/${seq}/" &


    # Run both commands sequentially in a single background process
    (
        python applications/generate_replica_class_color.py scene_id=$seq

        python -m applications.runner_dataset scene_id=$seq use_rerun=false

        # python sem_seg_eval.py scene_id=$seq

    ) &
    CHILD_PID=$!
    
    > "${metrics_dir}/memory.txt"
    > "${metrics_dir}/cpu.txt"
    > "${metrics_dir}/gpu.txt"
    
    echo "Started processing $seq (PID: $CHILD_PID)"
    
    while kill -0 $CHILD_PID 2>/dev/null; do
        timestamp=$(date +%s)
        mem_usage=$(get_memory_usage)
        cpu_usage=$(get_cpu_usage)
        gpu_metrics=$(get_gpu_metrics)
        
        echo "$timestamp $mem_usage" >> "${metrics_dir}/memory.txt"
        echo "$timestamp $cpu_usage" >> "${metrics_dir}/cpu.txt"
        echo "$timestamp $gpu_metrics" >> "${metrics_dir}/gpu.txt"
        
        update_metrics_json "$seq" "$start_time" "$metrics_dir"
        
        sleep 1
    done
    
    wait $CHILD_PID
    
    end_time=$(date +%s)
    processing_time=$((end_time - start_time))
    
    echo "=== Statistics for $seq ===" >> statistics_replica_mobilclip.txt
    echo "Processing time: $processing_time seconds" >> statistics_replica_mobilclip.txt
    
    mem_avg=$(awk '{ sum += $2 } END { print sum/NR }' "${metrics_dir}/memory.txt")
    mem_peak=$(awk 'BEGIN{max=0}{if($2>max)max=$2}END{print max}' "${metrics_dir}/memory.txt")
    echo "Memory usage (MB) - Avg: $mem_avg, Peak: $mem_peak" >> statistics_replica_mobilclip.txt
    
    cpu_avg=$(awk '{ sum += $2 } END { print sum/NR }' "${metrics_dir}/cpu.txt")
    cpu_peak=$(awk 'BEGIN{max=0}{if($2>max)max=$2}END{print max}' "${metrics_dir}/cpu.txt")
    echo "CPU usage (%) - Avg: $cpu_avg, Peak: $cpu_peak" >> statistics_replica_mobilclip.txt
    
    gpu_mem_avg=$(awk -F',' '{ sum += $1 } END { print sum/NR }' "${metrics_dir}/gpu.txt")
    gpu_mem_peak=$(awk -F',' 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "${metrics_dir}/gpu.txt")
    gpu_util_avg=$(awk -F',' '{ sum += $2 } END { print sum/NR }' "${metrics_dir}/gpu.txt")
    gpu_util_peak=$(awk -F',' 'BEGIN{max=0}{if($2>max)max=$2}END{print max}' "${metrics_dir}/gpu.txt")
    echo "GPU memory (MB) - Avg: $gpu_mem_avg, Peak: $gpu_mem_peak" >> statistics_replica_mobilclip.txt
    echo "GPU utilization (%) - Avg: $gpu_util_avg, Peak: $gpu_util_peak" >> statistics_replica_mobilclip.txt
    echo "" >> statistics_replica_mobilclip.txt
    
    CHILD_PID=""
    
    echo "Completed processing $seq"
done

# Mark processing as completed
echo "{\"currentSequence\": \"completed\"}" > metrics_new.json
merge_metrics_json

echo "All sequences processed. Statistics saved in statistics_replica_mobilclip.txt"
echo "View monitor.html for visualization of results"

