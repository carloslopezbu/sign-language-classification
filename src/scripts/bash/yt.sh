#!/bin/bash

# Verificar que se pase el archivo como argumento
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <archivo_con_ids>"
    exit 1
fi

FILE=$1

# Verificar que el archivo existe
if [ ! -f "$FILE" ]; then
    echo "Error: El archivo '$FILE' no existe."
    exit 1
fi

# Verificar que yt-dlp está instalado
if ! command -v yt-dlp &> /dev/null; then
    echo "Error: yt-dlp no está instalado."
    echo "Instálalo con: pip install yt-dlp"
    exit 1
fi

# Contador de éxitos y fallos
SUCCESS=0
FAILED=0

echo "==========================================================="
echo "Iniciando descarga de videos (SOLO VIDEO, SIN AUDIO)"
echo "==========================================================="

# Leer el archivo línea por línea
while IFS= read -r VIDEO_ID || [ -n "$VIDEO_ID" ]; do

    # Limpiar espacios en blanco y caracteres de retorno de carro
    VIDEO_ID=$(echo "$VIDEO_ID" | tr -d '\r' | xargs)

    # Saltar líneas vacías o comentarios
    if [ -z "$VIDEO_ID" ] || [[ "$VIDEO_ID" == \#* ]]; then
        continue
    fi

    echo ""
    echo "---------------------------------------------------"
    echo "Procesando ID: $VIDEO_ID"
    echo "---------------------------------------------------"

    # Descargar SOLO VIDEO (sin audio)
    # -f 'bv*[ext=mp4]/bv*/b': Formato flexible que encuentra video disponible
    # --remux-video mp4: Convierte a mp4 sin recodificar (más rápido)
    # -N 8: Usa 8 conexiones para descarga más rápida
    # -o '%(id)s.mp4': Guarda con el ID como nombre
    # --no-check-certificates: Evita problemas con certificados
    # --extractor-args "youtube:player_client=android": Usa cliente Android (evita el challenge)

    if yt-dlp \
        -f 'bv*[ext=mp4]/bv*/b' \
        --remux-video mp4 \
        --extractor-args "youtube:player_client=android" \
        -N 8 \
        --no-check-certificates \
        -o '%(id)s.mp4' \
        "https://www.youtube.com/watch?v=$VIDEO_ID"; then

        echo "✓ Descarga exitosa: $VIDEO_ID"
        ((SUCCESS++))
    else
        echo "✗ Error al descargar: $VIDEO_ID"
        ((FAILED++))
    fi

done < "$FILE"

echo ""
echo "==========================================================="
echo "Proceso finalizado"
echo "==========================================================="
echo "Videos descargados exitosamente: $SUCCESS"
echo "Videos con errores: $FAILED"
echo "==========================================================="
