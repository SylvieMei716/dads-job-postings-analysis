## Nov 14 update message
Please move the new `industries.csv` to your `data` folder.

## Installation

`python -m spacy download en_core_web_sm`
`!wget http://nlp.stanford.edu/data/glove.6B.zip`
`!unzip glove.6B.zip`

## Structure
```
Project/
├── README.md
├── data/
│   ├── companies.csv
│   ├── company_industries.csv
│   ├── employee_counts.csv
│   ├── gsearch_jobs.csv
│   ├── industries.csv
│   ├── verify_job_level_extraction.csv (generated in notebook)
│   └── dataframe_after_preprocessing.csv (generated in notebook)
├── final_project.ipynb
├── .gitignore (ignoring data folder)
├── README.md
├── requirements.txt
├── skill_db_relax_20.json (generated in notebook)
└── token_dist.json (generated in notebook)
```

## Pipiline
1. run all in `preprocessing.ipynb`, get the processed data
2. run all in `models.ipynb`