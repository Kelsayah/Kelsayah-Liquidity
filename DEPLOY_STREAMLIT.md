# Despliegue en Streamlit Community Cloud

La aplicación principal es `app.py` y las dependencias están declaradas en
`requirements.txt`.

## Secreto necesario

En **App settings → Secrets** añade:

```toml
FRED_API_KEY = "tu_clave_real"
```

Los archivos `.ENV`, la caché y los registros locales están excluidos del
repositorio. Si una fuente externa falla temporalmente, la caché se reconstruye
durante la ejecución de la aplicación.
