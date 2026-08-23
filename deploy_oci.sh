#!/bin/bash
# Script de deploy na OCI Compute (Ubuntu)
set -e

echo "=== Alura Agent Conciliação — Deploy na OCI ==="

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Edite o arquivo .env e adicione sua GOOGLE_API_KEY"
fi

sudo tee /etc/systemd/system/alura-agent.service > /dev/null <<EOF
[Unit]
Description=Alura Agent Conciliacao Bancaria
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable alura-agent
sudo systemctl start alura-agent

echo ""
echo "✅ Deploy concluído!"
echo "📌 Acesse: http://$(curl -s ifconfig.me):8501"
