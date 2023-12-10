# Spam Classifier Web Application

## Overview

This project is a web application built with Flask, incorporating a machine learning model for spam classification. Users can input text, and the application predicts whether the text is spam or not. The application also logs predictions to a MySQL database and provides a user interface for viewing prediction logs.


## Getting Started

### Local Development

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/spam-classifier.git
    cd spam-classifier
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application locally:

    ```bash
    python app.py
    ```

    The application will be accessible at [http://localhost:5000](http://localhost:5000).

### Docker Deployment

1. Build the Docker image:

    ```bash
    docker build -t spam-app:v1 .
    ```

2. Run the Docker container:

    ```bash
    docker run -p 5000:5000 spam-app:v1
    ```

    The application will be accessible at [http://localhost:5000](http://localhost:5000).

### Kubernetes Deployment

1. Apply the Kubernetes deployment and service YAML files:

    ```bash
    kubectl create --filename deployment.yaml
    kubectl create --filename service.yaml
    ```

2. Access the application:

    Get the external IP address:

    ```bash
    kubectl get services
    ```

    Access the application using the external IP address.

## Project Structure

- **app.py**: Flask application for spam classification.
- **index.html**: HTML template for the main page.
- **db_logs.html**: HTML template for viewing prediction logs.
- **requirements.txt**: List of Python dependencies.
- **Dockerfile**: Docker configuration for building the application image.
- **deployment.yaml**: Kubernetes deployment configuration.
- **service.yaml**: Kubernetes service configuration.

## Configuration

- **app.config['SQLALCHEMY_DATABASE_URI']**: Database URI for SQLAlchemy. Modify this for your database configuration.

## Dependencies

- Flask==3.0.0
- scikit-learn==1.3.2
- pandas==2.0.3
- torch==1.10.0+cpu
- Werkzeug==3.0.1
- urllib3==2.0.7
- Flask-SQLAlchemy==3.1.1
- PyMySQL==1.0.3
- SQLAlchemy==2.0.23
- mysql-connector-python==8.2.0

