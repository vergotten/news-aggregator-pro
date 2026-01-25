FROM ollama/ollama:latest

# Установить модели при сборке образа
RUN ollama serve & \
    sleep 5 && \
    ollama pull mistral:latest && \
    ollama pull llama3:latest && \
    pkill ollama

# Опционально: DeepSeek R1 20B (раскомментируйте если хотите)
# RUN ollama serve & \
#     sleep 5 && \
#     ollama pull deepseek-r1:20b && \
#     pkill ollama

EXPOSE 11434

CMD ["/bin/ollama", "serve"]
