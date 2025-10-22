from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import boto3
import json
import os

# Cargar variables de entorno
load_dotenv()

# Configuración de AWS
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")

# Crear cliente SageMaker Runtime
runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Inicializar FastAPI
app = FastAPI(title="SageMaker Endpoint API", version="1.0")


@app.post("/predict")
async def predict(request: Request):
    """
    Envía datos al endpoint de SageMaker y devuelve la predicción.
    """
    try:
        # Obtener el cuerpo JSON del request
        data = await request.json()

        # Enviar al endpoint de SageMaker
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(data),
        )

        # Leer respuesta del modelo
        result = response["Body"].read().decode("utf-8")
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))