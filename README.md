# pabd_cv
Predictive analytics practice repo for computer vision students

## План семинаров

1. Основы работы с bash. 
Система версионирования Git.
Модель разработки GitHub Flow. 
Настройка виртуальной среды python. 
Установка зависимостей, пакетные менеджеры. 
Продвинутые возможности языка python.  

**Результат:** fork репозитория, создание файла services/server_xxx.py , pull request в основной репозиторий.   

2. Структура ML проекта, шаблонизация cookiecutter ds. 
Требования к коду: codestyle, linters, formatters, function docs. 
Реализация минимального функционала классификации изображений. 
Тестирование с помощью unittest. 

**Результат:**  pull request c predict необученной модели.

3. Хранение данных в S3 хранилище. 
Версионирование данных с DVC. 
CLI python. 
Обучение модели. 

**Результат:** pull request c predict обученной модели.


4. Валидация модели. 
Добавление новых данных. 
Мониторинг метрик и производительности модели.  

**Результат:** скрипт валидации модели validate.py, деплой модели на препродакшн сервер.   

Запустить мою работу можно с помощью команды:
docker run -p 8888:8888 natashagushan/pabd:latest
