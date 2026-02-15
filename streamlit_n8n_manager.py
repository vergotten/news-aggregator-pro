# streamlit_n8n_manager.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
from src.application.services.n8n_service import N8nService
from src.infrastructure.n8n.n8n_client import N8nClient

# Настройка страницы
st.set_page_config(
    page_title="N8n Workflow Manager",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Инициализация сервиса
@st.cache_resource(ttl=300)  # Кешируем на 5 минут
def get_n8n_service():
    return N8nService()


# Синхронная обертка для асинхронных функций
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Загрузка данных
@st.cache_data(ttl=60)  # Кешируем на 1 минуту
def load_dashboard_data():
    service = get_n8n_service()
    return run_async(service.get_dashboard_data())


# Главный интерфейс
def main():
    st.title("⚙️ N8n Workflow Manager")
    st.markdown("Управление воркфлоу и расписаниями n8n")

    # Боковая панель
    with st.sidebar:
        st.header("Навигация")
        page = st.selectbox(
            "Выберите раздел",
            ["📊 Дашборд", "🔄 Воркфлоу", "⏰ Расписания", "📈 Исполнения"]
        )

        # Кнопка обновления
        if st.button("🔄 Обновить данные", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        data = load_dashboard_data()

    # Отображение страниц
    if page == "📊 Дашборд":
        show_dashboard(data)
    elif page == "🔄 Воркфлоу":
        show_workflows(data)
    elif page == "⏰ Расписания":
        show_schedules(data)
    elif page == "📈 Исполнения":
        show_executions(data)


def show_dashboard(data):
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего воркфлоу", data["total_workflows"])
    with col2:
        st.metric("Активных", data["active_workflows"])
    with col3:
        st.metric("Расписаний", data["total_schedules"])
    with col4:
        st.metric("Success Rate", f"{data['success_rate']}%")

    st.divider()

    # Графики
    col1, col2 = st.columns(2)

    with col1:
        # Статусы воркфлоу
        if data["workflows"]:
            status_counts = {}
            for wf in data["workflows"]:
                status = wf.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Статусы воркфлоу"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Последние исполнения
        if data["executions"]:
            exec_df = pd.DataFrame([
                {
                    "ID": ex.id[:8],
                    "Статус": ex.status.value,
                    "Время": ex.started_at.strftime("%H:%M") if ex.started_at else "—"
                }
                for ex in data["executions"][:10]
            ])
            st.dataframe(exec_df, use_container_width=True)


def show_workflows(data):
    st.subheader("🔄 Управление воркфлоу")

    if not data["workflows"]:
        st.info("Нет доступных воркфлоу")
        return

    # Таблица воркфлоу
    for wf in data["workflows"]:
        with st.expander(f"📄 {wf.name} ({wf.status.value})"):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**ID:** {wf.id}")
                st.write(f"**Описание:** {wf.description or '—'}")
                st.write(f"**Теги:** {', '.join(wf.tags) if wf.tags else '—'}")
                if wf.created_at:
                    st.write(f"**Создан:** {wf.created_at.strftime('%Y-%m-%d %H:%M')}")

            with col2:
                status_color = "🟢" if wf.status.value == "active" else "🔴"
                st.markdown(f"**Статус:** {status_color} {wf.status.value}")

            with col3:
                if st.button(f"{'Деактивировать' if wf.status.value == 'active' else 'Активировать'}",
                             key=f"toggle_{wf.id}"):
                    service = get_n8n_service()
                    success = run_async(service.toggle_workflow(wf.id))
                    if success:
                        st.success("✅ Статус изменен")
                        st.rerun()
                    else:
                        st.error("❌ Ошибка")

                if st.button("▶️ Выполнить", key=f"exec_{wf.id}"):
                    service = get_n8n_service()
                    exec_id = run_async(service.execute_workflow_manually(wf.id))
                    if exec_id:
                        st.success(f"✅ Запущено: {exec_id}")
                    else:
                        st.error("❌ Ошибка запуска")


def show_schedules(data):
    st.subheader("⏰ Управление расписаниями")

    # Создание нового расписания
    with st.expander("➕ Создать новое расписание"):
        with st.form("create_schedule"):
            workflows = data["workflows"]
            if workflows:
                workflow_options = {f"{wf.name} ({wf.id})": wf.id for wf in workflows}
                selected_wf = st.selectbox("Воркфлоу", list(workflow_options.keys()))

                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Название расписания")
                    cron = st.text_input("Cron выражение", value="0 0 * * *")
                with col2:
                    is_active = st.checkbox("Активно", value=True)
                    timezone = st.selectbox("Часовой пояс", ["UTC", "Europe/Moscow", "America/New_York"])

                if st.form_submit_button("Создать"):
                    schedule_data = {
                        "workflow_id": workflow_options[selected_wf],
                        "name": name,
                        "cron_expression": cron,
                        "is_active": is_active,
                        "timezone": timezone
                    }
                    service = get_n8n_service()
                    success = run_async(service.create_schedule(schedule_data))
                    if success:
                        st.success("✅ Расписание создано")
                        st.rerun()
                    else:
                        st.error("❌ Ошибка создания")
            else:
                st.warning("Нет доступных воркфлоу")

    # Список расписаний
    if data["schedules"]:
        st.divider()
        for schedule in data["schedules"]:
            with st.expander(f"⏰ {schedule.name} ({'Активно' if schedule.is_active else 'Неактивно'})"):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"**ID:** {schedule.id}")
                    st.write(f"**Cron:** `{schedule.cron_expression}`")
                    st.write(f"**Часовой пояс:** {schedule.timezone}")
                    if schedule.next_run:
                        st.write(f"**Следующий запуск:** {schedule.next_run.strftime('%Y-%m-%d %H:%M')}")

                with col2:
                    if st.button("✏️ Изменить", key=f"edit_{schedule.id}"):
                        st.session_state[f"edit_schedule_{schedule.id}"] = True

                with col3:
                    if st.button("🗑️ Удалить", key=f"del_{schedule.id}"):
                        service = get_n8n_service()
                        success = run_async(service.delete_schedule(schedule.id))
                        if success:
                            st.success("✅ Удалено")
                            st.rerun()
                        else:
                            st.error("❌ Ошибка удаления")

                # Форма редактирования
                if st.session_state.get(f"edit_schedule_{schedule.id}"):
                    with st.form(f"edit_form_{schedule.id}"):
                        new_name = st.text_input("Название", value=schedule.name)
                        new_cron = st.text_input("Cron", value=schedule.cron_expression)
                        new_active = st.checkbox("Активно", value=schedule.is_active)
                        new_tz = st.selectbox("Часовой пояс", ["UTC", "Europe/Moscow"],
                                              index=["UTC", "Europe/Moscow"].index(schedule.timezone))

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.form_submit_button("💾 Сохранить"):
                                schedule_data = {
                                    "name": new_name,
                                    "cron_expression": new_cron,
                                    "is_active": new_active,
                                    "timezone": new_tz
                                }
                                service = get_n8n_service()
                                success = run_async(service.update_schedule(schedule.id, schedule_data))
                                if success:
                                    st.success("✅ Сохранено")
                                    st.session_state[f"edit_schedule_{schedule.id}"] = False
                                    st.rerun()
                                else:
                                    st.error("❌ Ошибка сохранения")
                        with col_cancel:
                            if st.form_submit_button("Отмена"):
                                st.session_state[f"edit_schedule_{schedule.id}"] = False
                                st.rerun()
    else:
        st.info("Нет созданных расписаний")


def show_executions(data):
    st.subheader("📈 История выполнений")

    if not data["executions"]:
        st.info("Нет данных о выполнениях")
        return

    # Фильтры
    with st.expander("🔍 Фильтры"):
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "Статус",
                options=["running", "success", "error", "canceled", "waiting"],
                default=["success", "error"]
            )
        with col2:
            time_range = st.selectbox(
                "Период",
                ["Последние 24ч", "Последние 7 дней", "Последние 30 дней", "Все время"]
            )

    # Подготовка данных
    executions_data = []
    for ex in data["executions"]:
        if ex.started_at:
            executions_data.append({
                "ID": ex.id[:8],
                "Воркфлоу": ex.workflow_id[:8],
                "Статус": ex.status.value,
                "Начало": ex.started_at,
                "Завершение": ex.stopped_at,
                "Длительность": (ex.stopped_at - ex.started_at).total_seconds() if ex.stopped_at else None,
                "Режим": ex.mode,
                "Ошибка": ex.error[:50] + "..." if ex.error and len(ex.error) > 50 else ex.error
            })

    df = pd.DataFrame(executions_data)

    # Применение фильтров
    if status_filter:
        df = df[df["Статус"].isin(status_filter)]

    if time_range != "Все время":
        now = datetime.utcnow()
        if time_range == "Последние 24ч":
            df = df[df["Начало"] >= now - timedelta(days=1)]
        elif time_range == "Последние 7 дней":
            df = df[df["Начало"] >= now - timedelta(days=7)]
        elif time_range == "Последние 30 дней":
            df = df[df["Начало"] >= now - timedelta(days=30)]

    # Отображение
    st.dataframe(
        df,
        column_config={
            "Начало": st.column_config.DatetimeColumn("Начало", format="YYYY-MM-DD HH:mm:ss"),
            "Завершение": st.column_config.DatetimeColumn("Завершение", format="YYYY-MM-DD HH:mm:ss"),
            "Длительность": st.column_config.NumberColumn("Длительность (с)", format="%.2f")
        },
        use_container_width=True
    )

    # График выполнений по времени
    if not df.empty:
        st.divider()
        st.subheader("📊 График выполнений")

        df["Дата"] = df["Начало"].dt.date
        daily_counts = df.groupby("Дата").size().reset_index(name="Количество")

        fig = px.line(
            daily_counts,
            x="Дата",
            y="Количество",
            title="Количество выполнений по дням"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()