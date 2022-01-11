# RTEX@E

Explainable AI for image classification in radiology.


This repository is based on the work done in [this article](https://academic.oup.com/jamia/article/28/8/1651/6242739).

## How to run it

1. Install the requirements in `requirements.txt`
   ```
   pip install -r requirements.txt
   ```
2. Run the `explainability.py` script
   ```
   python explainability.py
   ```
3. Use the `load_mimic_data.py` script
   
The script needs a `mimic_config.json` file to run. The structure has to be as follows:

```{json}
{
  "physionet_url": "https://physionet.org/files/mimic-cxr/2.0.0/files",
  "physionet_user": "<YOUR USERNAME>",
  "physionet_pass": "<YOUR PASSWORD>",
  "pat_group": "<DESIRED PATIENT GROUP>",
  "patient_ids": [
    "p10000032",
    "p10000032"
    ]   
}
```


## Output
Based on the settings selected in the `explainability.py` script the generated plots will appear in the `plots` folder

## Examples

### Normal images

![](examples/normal/CXR456_output.png)

### Abnormal images

![](examples/abnormal/CXR1383_output.png)

### Tags

![](examples/tags/CXR471-degenerative%20change_output.png)

