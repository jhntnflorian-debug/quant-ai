# Plan de despliegue y presupuesto

Estado: plan. Nada de esto está contratado. Contratar cualquier servicio de pago requiere tu autorización.

## Separación de responsabilidades

| Función | Dónde | Por qué |
|---|---|---|
| Entrenamiento DL, RL y benchmarks | Google Colab (GPU) | Ya lo tienes; se paga por unidades solo cuando entrenas |
| Validación ML (LR + LightGBM) | CPU local con 2 hilos (111 s medidos) o Colab | No necesita GPU |
| Almacenamiento de modelos y reportes | Google Drive (`QUANT_AI_HOME`) | Persistente entre sesiones de Colab |
| Inferencia y ejecución paper 24/7 | VPS pequeño fuera de EE. UU. | Colab no sirve para procesos 24/7. Binance.com bloquea a EE. UU. según su lista de países prohibidos |
| Dashboard | El mismo VPS, detrás de un túnel SSH o con autenticación | Evita exponer el puerto 8501 a Internet |
| Observabilidad | `state/heartbeat.json`, logs y una alerta externa | Ver abajo |

El proceso de paper trading es liviano: un ciclo por hora de ~4 s (medido: 3.6 s), unos MB de datos y menos de 512 MB de RAM.

## Opciones de VPS (precios del 2026-11-08)

| Opción | Precio | Notas | Fuente |
|---|---|---|---|
| Oracle Cloud Always Free (Ampere A1) | 0 | Hasta 2 OCPU y 12 GB. Oracle puede reclamar instancias ociosas (CPU, red y memoria bajo 20 % durante 7 días). Con un ciclo por hora es un riesgo real | [Oracle](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm) |
| Hetzner CX23 (UE) | 5.99 EUR/mes sin IVA | 2 vCPU y 4 GB. La página marcaba "currently unavailable" al verificar | [Hetzner](https://www.hetzner.com/cloud/cost-optimized) |
| Fly.io shared-cpu-1x 512 MB | 3.69 USD/mes | Sin capa gratuita para cuentas nuevas | [Fly.io](https://docs.fly.io/about/pricing) |
| Railway Hobby | 5 USD/mes con 5 USD de uso incluido | Un proceso 24/7 puede exceder el uso incluido | [Railway](https://docs.railway.com/pricing/plans) |
| Google Cloud e2-micro gratis | 0 | Solo en regiones de EE. UU. (us-west1, us-central1, us-east1): **no apto** para Binance.com | [GCP](https://docs.cloud.google.com/free/docs/compute-free-tier) |
| PC propio | 0 | Depende de que el equipo esté encendido; útil para pruebas cortas | — |

**Recomendación**: empezar en el PC propio con `run_paper.py --once` programado cada hora desde el Programador de tareas de Windows, durante 1 o 2 semanas para validar estabilidad. Después, un VPS de unos 4 a 6 USD/mes en la UE (Hetzner o Fly.io). Presupuesto mensual estimado: **0 USD** en la fase inicial y **4 a 7 USD** con VPS, más las unidades de Colab que decidas gastar. El costo por hora de cada GPU lo mide el notebook con el valor que muestra tu cuenta.

## Pasos en un VPS Linux (cuando lo autorices)

```bash
git clone <tu-repo> quant-ai && cd quant-ai
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e .
python scripts/download_data.py
# copiar models/registry desde Drive (rclone o scp)
docker compose -f docker/compose.yaml up -d   # o un servicio systemd con run_paper.py
```

## Reentrenamiento y promoción controlada

1. Reentrenar (Colab, mensual o trimestral): `run_experiments.py --register` crea una versión nueva en `models/registry/` con estado **candidato**.
2. La puerta de promoción (`promotion_gate` en `training/experiment.py`) exige, en el walk-forward: Sharpe del ensemble mayor que el de comprar y mantener, retorno positivo con costes ×2, drawdown no peor que comprar y mantener, y un IC95 % del Sharpe que excluya 0.
3. Solo si pasa la puerta se escribe `models/registry/ACTIVE`. Ningún proceso reemplaza el modelo activo de forma automática sin pasar esa puerta.
4. Recomendado antes de promover a dinero real: al menos 4 semanas de paper con el candidato, comparando el desvío entre paper y backtest en el dashboard.

## Observabilidad mínima

- `state/heartbeat.json`: hora del último ciclo, estado (`ok`, `error`, `kill_switch`, `emergency`) y antigüedad de los datos.
- Alerta sugerida (sin costo): un cron que revise que `last_cycle_at` tenga menos de 2 h y envíe un correo o un mensaje de Telegram si no es así. No está implementado.
- Logs del proceso con rotación (systemd o Docker).
