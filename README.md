# Ultimate Chatbot for Education - Core Model

_This project was bootstrapped with **Build with model SVM and LogisticRegression**_

## Deployment in development mode

### `Install anaconda`

-   Download anaconda and install it by following this [link](https://www.anaconda.com/products/individual)
-   Create virtual environment with python 3.7

### `Install package`

-   Open your favorite command line
-   Move to the folder containing the project and run command

```
pip install flask flask_cors
pip install -r requirements.txt
```

### `Run app`

-   Run command

```
python main.py
```

### Finally, access to localhost:8080

## Deployment in production mode on Ubuntu 20.04 server with Flask, Gunicorn and Nginx

### Install ubuntu and setup init

Following this [link](https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-20-04)

### `Install and create a Python Virtual Environment`

-   Install package

```
sudo apt update

sudo apt install python3-pip python3-dev build-essential libssl-dev libffi-dev
python3-setuptools

sudo apt install python3-venv
```

-   Clone app

```
git clone https://github.com/Fail-Capstone/ultimate_chatbot_core.git chatbotcore

cd ~/chatbotcore
```

-   Create virtual environment

```
python3 -m venv chatbotenv

source chatbotenv/bin/activate
```

-   Setup Flask and run test

```
pip install wheel

pip install gunicorn flask

sudo ufw allow 8080

python main.py
```

-   Run server with gunicorn

```
gunicorn --bind 0.0.0.0:5000 wsgi:app --daemon
```

### Setup Nginx Server Blocks

Following this [link](https://www.digitalocean.com/community/tutorials/how-to-set-up-nginx-server-blocks-virtual-hosts-on-ubuntu-16-04)

-   enable ufw
-   allow port 8080
-   restart server nginx

### Finally, access to IP address server

## Introduction for use

- Check app is running on

```json
GET https://your_server_ip:8080
```

- Get answer

```json
POST https://your_server_ip:8080
Content-type: application/json
{
    "question": "{câu hỏi}"
}
```
- Train model

```json
GET https://your_server_ip:8080/train
```
