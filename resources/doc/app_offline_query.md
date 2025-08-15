# Application: Offline Map Query

<p align="center">
    <img src="../image/query.jpg" width="90%">
</p>


## Data Download

We provide two prebuilt concrete maps for offline query experiments: 1. Replica `room_0` ; 2. Record3D (iPhone)

You can download the data here:
🔗 [OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/Eg4WTY_fC3tBkeiv2vU7dc4BUGEVGjQu5FGhWk_ZlNJ3tQ?e=hVRkgY)
🔗[Google Drive](https://drive.google.com/drive/folders/1y5ZCd5cvIUKR7XAS_5m7KCeLaKZEv7ok?usp=sharing)


### Example: `replica_room_0/` Directory Structure
After downloading and unzipping the dataset, your directory should look like this:
```
replica_room_0/
└── map/
    ├── *.pkl                     # Objects Representation
    ├── layout.pcd                # Point cloud layout of the scene
    └── viewpoint.json            # Viewpoint metadata
```

## Querying
Once the example data is downloaded, you can start querying the map.
### Configure the input map directory
Edit the config file: `config/query_config.yaml`
```
map_dir: "<PATH TO MAP>"

# Example for Replica:
# map_dir: "<YOUR_PATH>/replica_room_0/map"

# Example for iPhone:
# map_dir: "<YOUR_PATH>/iphone_table/map"
```
### Run the query application

```
python -m applications.offline_local_map_query
```

## Usage
The usage instructions will be printed in the terminal when the application starts.

| Key | Action                                     |
| --- | ------------------------------------------ |
| `Q` | Quit the application                       |
| `R` | Show point cloud with RGB colors           |
| `C` | Show point cloud with semantic colors      |
| `F` | Enter a query to find top-matching objects |
| `H` | Display the help message                   |
| `N` | Highlight queried objects                  |
| `M` | Similarity Map of queried objects          |
| `S` | Save the current viewpoint                 |

> Press the corresponding key to perform the action in the Open3D viewer.