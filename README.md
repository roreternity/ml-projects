# ML Portfolio — Kaggle и университетская практика

Коллекция проектов по машинному обучению: классификация, регрессия, компьютерное зрение, NLP, прогнозирование временных рядов, веб-скрапинг и дообучение LLM.

В каждой папке проекта — свой `README.md` с описанием задачи и `*_notebook.ipynb` с полным разбором на русском языке (код и вывод — на английском).

---

## Проекты

### Соревнования Kaggle

| Проект | Задача | Модель | Результат на Kaggle |
|---|---|---|---|
| [Titanic](./Titanic/) | Бинарная классификация — предсказание выживания | Random Forest | 0.77990 (accuracy) |
| [House Prices](./House_Prices/) | Регрессия — предсказание цены дома | XGBoost | 0.13786 (RMSE, log scale) |
| [Spaceship Titanic](./Spaceship/) | Бинарная классификация с категориальными признаками | CatBoost | 0.79822 (accuracy) |
| [Digit Recognizer](./Digit_Recognizer/) | Классификация изображений (MNIST) | CNN (PyTorch) | 0.98225 (accuracy) |
| [Dogs vs Cats](./Dogs_Cats/) | Бинарная классификация изображений | CNN (PyTorch) | 6.94102 (log loss) |
| [Stellar Object Classification](./Stellar_Class/) | Многоклассовая классификация (SDSS) | Random Forest | 0.94275 (public) |
| [Student Health Classification](./Student_Health/) | Многоклассовая классификация, несбалансированные классы | Random Forest | 0.85819 (public) |

### Университетская практика

| Проект | Задача | Метод | Результат |
|---|---|---|---|
| [Guardian Data Collection](./News_Parser/) | Сбор данных из веб-источников через API | Guardian Content API | 5000 статей, 12 признаков |
| [Support Ticket Classifier](./Ticket_Classifier/) | Многоклассовая классификация текста | TF-IDF + LinearSVC | 86.79% accuracy (8 классов) |
| [Time Series Forecasting](./Energy_Forecasting/) | Прогноз по 61 временному ряду на 36 месяцев | Сезонный mean-delta | SMAPE 18.35% против 21.64% baseline |
| [Least Squares & Linear Regression](./Studying/) | Аппроксимация кривых, основы регрессии | numpy / sklearn / scipy | — |

### Другое

| Проект | Описание |
|---|---|
| [DojoTask](./DojoTask/) | Перевод EN→RU: дообучение Gemma методом QLoRA |
| [MII](./MII/) | Скрипты веб-скрапинга (вакансии HH.ru, товары Wildberries) |
| [SHAD_Handbook](./SHAD_Handbook/) | Материалы курса по машинному обучению |

---

## Другие репозитории портфолио

| Репозиторий | Описание |
|---|---|
| [Yandex ML Cup](https://github.com/roreternity/yandex_ml_cup) | Решения трёх треков Yandex ML Cup: адаптивный решатель головоломок, синтез новых ракурсов по видео, LLM для ответов на школьные вопросы |
| [Yandex SAR 2026](https://github.com/roreternity/yandex-sar-2026) | Задачи Yandex SAR: A/B-тестирование, дешифровка, деревья решений и др. |
| [StoRM (Team Risk Optimizer)](https://github.com/roreternity/team-risk-optimizer) | Проект по анализу и оптимизации рисков в команде |

---

## Данные

Каждый ноутбук ожидает соответствующий датасет Kaggle в `/kaggle/input/...` при запуске на Kaggle, либо CSV-файлы рядом со скриптом при локальном запуске.

## Стек

Python · pandas · scikit-learn · XGBoost · CatBoost · PyTorch · NumPy
