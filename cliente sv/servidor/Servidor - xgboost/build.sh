#!/bin/bash

CONTAINER_NAME=classificador-especializada
IMAGE_NAME=especializada-xgb

echo "🔨 Buildando a imagem..."
docker build -t $IMAGE_NAME .

echo "➡️ Parando container antigo (se existir)..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null


echo "🚀 Iniciando o container..."
docker run -d \
 --name $CONTAINER_NAME \
 --env-file .env \
 --restart always \
 -p 5001:5001 \
 -v "/opt/apps/files/classificador_especializadas/DirTrein:/app/DirTrein" \
 $IMAGE_NAME \

echo "✅ Container '$CONTAINER_NAME' rodando em http://localhost:5001"