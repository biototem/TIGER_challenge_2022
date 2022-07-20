# Regions_Segmentation

This subdirectory provides training  scripts for segmentation in [**TIGER**](https://tiger.grand-challenge.org/Data/)

## Process introduction

The startup code of the process is **./start_{process_name}.py**, and its corresponding implementation code is always placed in **./process/{name}**  . When starting the project process, please **make sure that the working directory is the project root directory** .

This project includes three sets of main processes and one set of independent processes, which are described as follows:

### pre-build process

The main process for loading data source information, startup name `process_name = 'build'`, and execution code is:`python ./start_build.py`

This process will generate important data that will be used in the training process `./cache`under . Since this project is a competition project, the preprocessing process is relatively simple.`source.lib`

### training process

The main process, used to obtain model weight, startup name `process_name = 'train'`, and execution code is:`python ./start_train.py`

The process will generate the training weight of each epoch and saved them at **./output/{output.target}** . And the `output.target` is the configuration item in **./CONFIG.yaml**, and now it is generally trained with 20 epoches.

### evaluation process

The  script to get the best model weight, start the name `process_name = 'evalute’`, execute the code is `python ./start_evaluate.py`and`python ./main/start_metric_visual.py`

`start_evaluate`The evaluation document will be generated under the target path and stored in Table format.

## Startup Instructions

In order to run the code for this project, you always need to ensure two things:

1. `./Segmentation`in path. This is because all import syntax in this project `./Segmentation` is rooted `python ./Segmentation/start_{}.py`in , and python will do this automatically when you execute , which is why all startup code is placed in `./Segmentation` .
2. `sysArg['workspace']`point to the root directory. This is because almost all code in the project strongly relies on the basic module `join('~/{relative_path}')`syntax , which always translates '~' to the directory specified by workspace. `python ./Segmentation/start_{}.py`The basic module automatically recognizes your current working directory when you execute under 'workspace' , so this is the recommended way to execute.

The following may be helpful when other intervening factors prevent you from executing code in `python ./Segmentation/start_{}.py`the manner:

1. In your IDE, `./main`set to Source Folders
2. Every time you run a script, you always specify a parameter `workspace={path}`, which you can set in the IDE's Run Configuration -> Templates, so you can still run the script in a familiar way.

## Data list

Considering the complexity of the data source, I made the relevant data files in a semi-manual way and stored them in json file. The data grouping basis during actual training is constrained by this file. The structre of json is as follows:

```python
{
    "TCGA-LL-A740-01Z-00-DX1.757D94A5-EF0F-4A0E-99A9-8809B66438DA_[19720, 12042, 21394, 13711]": {
        "folder": 1,
        "width": 1674,
        "height": 1669,
        "image": "/YOUE_DIR/wsirois/roi-level-annotations/tissue-bcss/images/TCGA-LL-A740-01Z-00-DX1.757D94A5-EF0F-4A0E-99A9-8809B66438DA_[19720, 12042, 21394, 13711].png",
        "label": "/YOUE_DIR/wsirois/roi-level-annotations/tissue-bcss/masks/TCGA-LL-A740-01Z-00-DX1.757D94A5-EF0F-4A0E-99A9-8809B66438DA_[19720, 12042, 21394, 13711].png",
        "available": true,
        "tag": "NORMAL",
        "group": ""
    },
     "TCGA-A7-A4SE-01Z-00-DX1.16BC8401-E40E-4A1A-9BD9-12735C9AE3F6_[25267, 5661, 26102, 6416]": {
        "folder": 1,
        "width": 835,
        "height": 755,
        "image": "/YOUE_DIR/wsirois/roi-level-annotations/tissue-bcss/images/TCGA-A7-A4SE-01Z-00-DX1.16BC8401-E40E-4A1A-9BD9-12735C9AE3F6_[25267, 5661, 26102, 6416].png",
        "label": "/YOUE_DIR/wsirois/roi-level-annotations/tissue-bcss/masks/TCGA-A7-A4SE-01Z-00-DX1.16BC8401-E40E-4A1A-9BD9-12735C9AE3F6_[25267, 5661, 26102, 6416].png",
        "available": false,
        "tag": "SMALL",
        "group": ""
    },
# .........
}
```

The data grouping method is as follows: 70% of the files in folder 1 are used as the training set and 30% as the validation set, and are divided in units of hospitals, and half of the files in folder 3 are used as the training set and half as the test set. Folder 2 is all used as a validation set.
