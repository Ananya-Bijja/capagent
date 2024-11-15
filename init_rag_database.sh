if [ -d "./chroma_db" ]; then
    rm -rf ./chroma_db
fi

python embedding.py
