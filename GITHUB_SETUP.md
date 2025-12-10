# Инструкция по публикации на GitHub

## Шаг 1: Настройка Git (если еще не настроено)

Выполните следующие команды, заменив на свои данные:

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "ваш-email@example.com"
```

## Шаг 2: Создание коммита

```bash
git commit -m "Initial commit: SVM Path Planner"
```

## Шаг 3: Создание репозитория на GitHub

1. Перейдите на https://github.com
2. Нажмите кнопку "+" в правом верхнем углу
3. Выберите "New repository"
4. Заполните:
   - Repository name: `svm-path-planner` (или другое имя)
   - Description: "SVM-based path planning system"
   - Выберите Public или Private
   - НЕ создавайте README, .gitignore или license (они уже есть)
5. Нажмите "Create repository"

## Шаг 4: Подключение к GitHub и отправка кода

После создания репозитория GitHub покажет инструкции. Выполните:

```bash
git remote add origin https://github.com/ВАШ-USERNAME/svm-path-planner.git
git branch -M main
git push -u origin main
```

Если вы используете SSH:

```bash
git remote add origin git@github.com:ВАШ-USERNAME/svm-path-planner.git
git branch -M main
git push -u origin main
```

## Альтернативный способ через GitHub CLI (если установлен)

```bash
gh repo create svm-path-planner --public --source=. --remote=origin --push
```

