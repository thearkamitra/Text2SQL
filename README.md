# Text2SQL Project

This repository contains the codebase for the Text2SQL project.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Text2SQL
```

### 2. Create a Virtual Environment
The python version used in this project is 3.10.12. Ensure you have it installed on your system.

```bash
For this project, it is recommended to use a virtual environment to manage dependencies. You can create one using the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Configure Environment Variables

There are some environment variables that need to be set for the project to run correctly. You can copy the example environment file and edit it:

Edit `.env` and fill in the required values:

```
# Example .env file
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
```

Refer to the project documentation for details on each variable.

### 5. Run the Project
From the root directory of the project, set the `PYTHONPATH` to include the current directory:
```bash
export PYTHONPATH=.
```
Then to obtain the scores, run the following command:

```bash
python -m eval/evaluate_non_langchain.py
```

Follow the instructions in the project documentation to start the application.

