# Snow identification

Project that predicts whether a patch of land is snow free or not

The method is based on the mean grayscaled pixel value around the center of the image. We take a box of (by default 25px by 25px) around the center point of the image, 
predict the mean grayscaled pixel value and compare it to a threshold. 

If the value is lower than 0.33, we consider the image to be snow free. 

If the value is from 0.33 to 0.66, we consider the image to be partially snow covered.

If the value is higher than 0.66, we consider the image to be snow covered.

# Prerequisites 

To start the project, we need to create a table in the `sqlite3` database.

To do so, run the following command: 

```bash
python -m init_db
```

# Virtual env 

To create the virtual env using anaconda `env.yml` file, run the following command: 

```bash
conda env create -f env.yml
```

To update the env, use the command: 

```bash
conda env update -f env.yml
```

# Project structure 

All the images should be put into the `input/` directory. 

The `output/` directory will contain a json with the image name and the result of the prediction. 