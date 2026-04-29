# Лабораторная работа №4

## Общие сведения
Лабораторная работа сделана с использованием [гугл диска](https://drive.google.com/drive/folders/1gf_a8RurgG6oWjwbV_fwNvIHqGbLwZpE?usp=sharing) в качестве хранилища датасетов в dvc

Мы использовали датасет titanic, который был описан в задании. Все его модификации также сделаны по шагам из задания.

Из-за того, что напрямую к гугл диску больше нельзя подключиться с помощью dvc, нужно сделать подключение к нему с помощью Google Cloud project([туториал из доков dvc](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended))

При подключении к диску мы должны передать user-id и user-secret из проекта, который мы заранее создаем в Google Cloud project

## Запуск
1. Склонируйте репозиторий
```
git clone https://github.com/devorkyan/mlops_practice.git
```
2. Создайте временное окружение и активируйте его
```
python -m venv .venv
.venv\Scripts\activate
```
3. Установите зависимости
```
pip install dvc dvc-gdrive pandas catboost scikit-learn
```
## Пошаговое выполнение лабораторной работы
1. Инициализируем dvc и коммитим изменения
```
dvc init
git add .dvc .dvcignore
git commit -m "init dvc"
``` 
2. Загружаем датасет titanic.csv, прописываем необходимые данные для подключения гугл диска к dvc, загружаем датасет в dvc и коммитим изменения
```
# загружаем датасет через catboost.datasets
python lab4/create_data.py

# прописываем необходимые данные для подключения гугл диска к dvc
dvc remote add --default myremote gdrive://your_shared_gdrive_id
dvc remote modify myremote gdrive_client_id 'client-id'
dvc remote modify myremote gdrive_client_secret 'client-secret'

# добавляем датасет в dvc и коммитим изменения
dvc add lab4/data
git add lab4/.gitignore lab4/data.dvc
git commit "lab4: created 1st version of titanic dataset"
dvc push
```
3. Делаем первое изменение датасета, оставляя колонки "Pclass", "Sex", "Age"
```
python lab4/crop_dataset.py

dvc add lab4/data/titanic.csv
git add 'lab4\data.dvc'
dvc push
```
4. Заполняем пропуски в поле "Age" средним значением и сохраняем новый датасет
```
python lab4/fillna.py

dvc add lab4/data/titanic_modified.csv
dvc push
```
5. Добавляем 2 новых признака по полу, используя one-hot-encoding
```
python lab4/titanic_oneshot.py

dvc add lab4/data/titanic_modified.csv
dvc push
```
## Возвращение конкретной версии датасета
1. Посмотрим все коммиты с их хешами
```
git log --oneline
```
2. К примеру, посмотрим на вторую версию датасета, в котором мы оставили только поля "Pclass", "Sex", "Age"
```
git checkout a856af2
dvc checkout
```
3. Чтобы вернуться к последней версии датасета
```
git checkout main
dvc checkout
```
