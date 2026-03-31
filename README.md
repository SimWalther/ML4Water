# ML4Water

ML4Water is a collaboration between a data science team at [HES-SO](https://www.hes-so.ch/en/homepage) and an environmental science team at [La Maison de la Rivière](https://www.maisondelariviere.ch/).

Through the application of machine learning, this project quantifies the relationship between riparian vegetation and stream temperatures. We utilize explainable AI (XAI) techniques to analyze how vegetation influences thermal dynamics and to pinpoint target areas where reforestation would be most effective for temperature regulation.

To achieve this, we developed a method to represent upstream riparian vegetation as structured image data, mapping vegetation buffers by width and distance relative to the stream. We then designed a slope-dependent attention mechanism that allows the model to dynamically focus on specific segments of this "vegetation image." This architecture enables us to determine which vegetation widths and upstream distances have the most significant impact on daily maximum water temperatures at a given station.

## Research Articles

We have authored two research articles detailing our methodology and findings (currently in the publication process):

* Predicting Stream Water Temperature: A Data-Driven Approach Highlighting the Impact of Riparian Vegetation
* Interpretable Machine Learning for Prioritizing Riparian Reforestation: A Case Study in a Swiss River

## Data Access

The datasets used in this study are publicly available on [Zenodo](https://zenodo.org/records/19236360).

## Usage

### Requirements
Ensure you are working within a Python environment, then install the dependencies via:

```bash
pip install -r requirements.txt
```

### Workflow
All code necessary is contained within the `utils/` directory. These functions are utilized in the primary notebook:

* `training_model_final.ipynb`

## Funding

This work was funded by a research grant from the [HES-SO](https://www.hes-so.ch/en/homepage). We gratefully acknowledge their financial contribution to this project.
