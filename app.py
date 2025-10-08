"""
AI Fitness Coach - Aplicação Streamlit
=====================================

Esta aplicação foi construída a partir de um notebook com o objetivo de gerar
planos de treino de musculação personalizados. O caderno original utilizava uma
arquitetura complexa baseada em LangGraph, integrando várias ferramentas externas e um modelo
de linguagem da Groq. Neste ambiente simplificado não é possível instalar essas
dependências ou fazer chamadas de rede para modelos de IA, portanto foi implementada uma
alternativa leve e auto‑contida que segue os mesmos princípios gerais: coletar dados do
usuário, realizar alguns cálculos básicos, consultar pesquisas na web quando possível e
gerar um plano de treino coerente com 12 sessões ao todo.

O código abaixo pode ser executado com `streamlit run app.py` em um ambiente onde a
biblioteca `streamlit` esteja instalada e onde o acesso à internet para consultas HTTP
esteja habilitado. Se a internet não estiver disponível, a função de pesquisa retornará
resultados vazios, mas o plano de treino continuará sendo gerado normalmente.
"""

import json
import math
import random
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


def normalizar_objetivo(goal: str) -> str:
    """Normaliza a string do objetivo para minúsculo e remove espaços adicionais."""
    return goal.strip().lower() if goal else ""


def normalizar_experiencia(level: str) -> str:
    """Normaliza o nível de experiência."""
    return level.strip().lower() if level else ""


def search_web(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Realiza uma busca simples usando a API pública do DuckDuckGo.

    A função tenta acessar `https://api.duckduckgo.com` que retorna um JSON contendo
    tópicos relacionados. Caso a requisição falhe (por exemplo, por falta de acesso
    à internet), a função retorna uma lista vazia com um item de erro para que o
    usuário tenha ciência do problema.

    Parameters
    ----------
    query : str
        A frase de busca enviada para a API.
    max_results : int
        Número máximo de resultados retornados.

    Returns
    -------
    List[Dict[str, str]]
        Uma lista de dicionários com campos 'title', 'snippet' e 'url'.
    """
    results: List[Dict[str, str]] = []
    if not query:
        return results
    try:
        # Utiliza a API de consulta da DuckDuckGo. O parâmetro `no_redirect` evita
        # redirecionamentos e `no_html` remove tags HTML do resultado.
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # A API retorna uma lista de tópicos relacionados em "RelatedTopics". Cada tópico
        # pode conter sub‑listas em "Topics"; percorremos recursivamente para coletar
        # títulos e URLs.
        def extract_topics(items):
            for item in items:
                if isinstance(item, dict):
                    # Elementos com campo 'Text' são resultados diretos
                    if 'Text' in item:
                        title = item.get('Text', '')
                        url = item.get('FirstURL', '')
                        yield {
                            "title": title,
                            "snippet": "",
                            "url": url,
                        }
                    # Em alguns casos há sub tópicos agrupados em "Topics"
                    if 'Topics' in item and isinstance(item['Topics'], list):
                        for sub in extract_topics(item['Topics']):
                            yield sub
        topics = list(extract_topics(data.get('RelatedTopics', [])))
        results.extend(topics[:max_results])
    except Exception as exc:
        # Caso a API falhe (por falta de conexão ou qualquer outra razão), retorna
        # um único registro informando o erro.
        results.append(
            {
                "title": "Erro na pesquisa",
                "snippet": f"Falha ao consultar a API de busca: {exc}",
                "url": "",
            }
        )
    return results


def ferramenta_matematica_treino(
    age: int,
    weight_kg: float,
    training_frequency: int,
    primary_goal: str,
    height_cm: Optional[float] = None,
    experience_level: str = "iniciante",
) -> Dict[str, Any]:
    """Calcula métricas básicas como calorias estimadas e rep ranges.

    Esta função replica a lógica apresentada no notebook original. Ela estima o
    gasto calórico por sessão e semanal, calcula o IMC quando a altura é
    fornecida, e determina sugestões de faixa de repetição, intensidade e
    volume conforme o objetivo e o nível de experiência.

    Returns
    -------
    dict
        Dicionário contendo métricas analíticas.
    """
    goal_key = normalizar_objetivo(primary_goal)
    # Parâmetros basais por objetivo
    goal_presets = {
        "hipertrofia": {
            "met": 6.0,
            "duration_hours": 1.2,
            "rep_range": "6-12",
            "intensity": "70-80% 1RM",
        },
        "emagrecimento": {
            "met": 5.5,
            "duration_hours": 1.15,
            "rep_range": "12-15",
            "intensity": "Circuitos com pausa curta",
        },
        "forca": {
            "met": 6.8,
            "duration_hours": 1.25,
            "rep_range": "3-6",
            "intensity": "80-90% 1RM",
        },
        "condicionamento": {
            "met": 7.2,
            "duration_hours": 1.05,
            "rep_range": "10-15",
            "intensity": "Intervalos moderados",
        },
    }
    preset = goal_presets.get(goal_key, goal_presets["hipertrofia"])
    bmi = None
    if height_cm:
        height_m = height_cm / 100.0
        bmi = round(weight_kg / (height_m ** 2), 2)
    session_calories = round(weight_kg * preset["met"] * preset["duration_hours"], 1)
    weekly_calories = round(session_calories * training_frequency, 1)
    volume_table = {
        "iniciante": "14-18 séries por grupamento",
        "intermediario": "18-22 séries por grupamento",
        "avancado": "22-26 séries por grupamento",
    }
    volume = volume_table.get(normalizar_experiencia(experience_level), "18-22 séries por grupamento")
    return {
        "goal": goal_key,
        "estimated_session_calories": session_calories,
        "estimated_weekly_calories": weekly_calories,
        "bmi": bmi,
        "recommended_rep_range": preset["rep_range"],
        "recommended_intensity": preset["intensity"],
        "volume_per_session": volume,
        "sessions_per_week": training_frequency,
        "experience_level": normalizar_experiencia(experience_level),
    }


def generate_plan(user_profile: Dict[str, Any], analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Cria um plano de treino estruturado contendo 12 sessões.

    A função constrói uma estrutura semelhante à especificada no notebook: uma
    visão geral (overview), diretrizes gerais, uma lista de semanas de treino
    contendo sessões detalhadas com exercícios, e seções de recuperação,
    nutrição e progressão. A lógica foi simplificada para gerar treinos
    plausíveis sem depender de modelos de linguagem externos. Os exercícios
    utilizados são escolhidos de listas predefinidas e adequadas ao objetivo.

    Parameters
    ----------
    user_profile : dict
        Dicionário contendo os dados normalizados do usuário.
    analytics : dict
        Métricas analíticas calculadas anteriormente.

    Returns
    -------
    dict
        Estrutura de plano de treino pronta para ser convertida em Markdown.
    """
    goal = user_profile["primary_goal"]
    freq = user_profile["training_frequency"]
    total_sessions = 12
    weeks = (total_sessions + freq - 1) // freq
    sessions_generated = 0
    training_weeks: List[Dict[str, Any]] = []
    # Dicionários de exercícios por grupamento
    exercise_dict = {
        "chest": [
            "Supino reto com barra",
            "Supino inclinado com halteres",
            "Crucifixo na máquina",
            "Peck deck",
            "Flexões",
        ],
        "triceps": [
            "Tríceps pulley",
            "Tríceps testa",
            "Mergulho banco",
            "Tríceps corda",
            "Kickback",
        ],
        "back": [
            "Puxada frontal",
            "Remada curvada com barra",
            "Remada unilateral com haltere",
            "Pulldown",
            "Levantamento terra",
        ],
        "biceps": [
            "Rosca direta com barra",
            "Rosca alternada",
            "Rosca martelo",
            "Rosca concentrada",
            "Rosca Scott",
        ],
        "legs": [
            "Agachamento livre",
            "Leg press",
            "Cadeira extensora",
            "Cadeira flexora",
            "Afundo com halteres",
            "Panturrilha em pé",
        ],
        "shoulders": [
            "Desenvolvimento com barra",
            "Desenvolvimento com halteres",
            "Elevação lateral",
            "Elevação frontal",
            "Remada alta",
        ],
        "glutes": [
            "Agachamento sumô",
            "Peso morto stiff",
            "Glute bridge",
            "Cadeira abdutora",
            "Elevação de quadril",
        ],
        "abs": [
            "Prancha",
            "Abdominal supra",
            "Elevação de pernas",
            "Abdominal oblíquo",
            "Prancha lateral",
        ],
    }
    # Exercícios de cardio/condicionamento utilizados em emagrecimento e condicionamento
    cardio_exercises = [
        "Burpees",
        "Mountain climbers",
        "Agachamento com salto",
        "Polichinelos",
        "Corrida estacionária",
        "Pular corda",
        "Kettlebell swing",
        "Clean and press com halteres leves",
        "Flexões",
        "Abdominal bicicleta",
    ]
    # Estruturas de divisões para cada objetivo
    hypertrophy_splits = [
        {
            "name": "Treino A - Peito e Tríceps",
            "muscles": ["chest", "triceps"],
            "focus": "Peito e Tríceps",
        },
        {
            "name": "Treino B - Costas e Bíceps",
            "muscles": ["back", "biceps"],
            "focus": "Costas e Bíceps",
        },
        {
            "name": "Treino C - Pernas e Ombros",
            "muscles": ["legs", "shoulders"],
            "focus": "Pernas e Ombros",
        },
        {
            "name": "Treino D - Peito e Costas",
            "muscles": ["chest", "back"],
            "focus": "Peito e Costas",
        },
        {
            "name": "Treino E - Pernas e Glúteos",
            "muscles": ["legs", "glutes"],
            "focus": "Pernas e Glúteos",
        },
        {
            "name": "Treino F - Ombros e Braços",
            "muscles": ["shoulders", "biceps", "triceps"],
            "focus": "Ombros e Braços",
        },
    ]
    strength_splits = [
        {
            "name": "Treino A - Agachamento",
            "muscles": ["legs", "glutes", "back"],
            "focus": "Força em Agachamento",
        },
        {
            "name": "Treino B - Supino e Press",
            "muscles": ["chest", "shoulders", "triceps"],
            "focus": "Força em Supino",
        },
        {
            "name": "Treino C - Deadlift",
            "muscles": ["legs", "back"],
            "focus": "Força em Deadlift",
        },
        {
            "name": "Treino D - Full Body",
            "muscles": ["legs", "back", "chest", "shoulders"],
            "focus": "Força Total",
        },
    ]
    conditioning_splits = [
        {"name": "Circuito HIIT", "muscles": [], "focus": "Alta intensidade"},
        {
            "name": "Circuito de Resistência",
            "muscles": [],
            "focus": "Resistência Muscular",
        },
        {
            "name": "Circuito Cardio e Força",
            "muscles": [],
            "focus": "Cardio e Força",
        },
        {"name": "Circuito Funcional", "muscles": [], "focus": "Funcional"},
    ]
    emagrecimento_splits = [
        {"name": "Circuito A", "muscles": [], "focus": "Circuito"},
        {"name": "Circuito B", "muscles": [], "focus": "Circuito"},
        {"name": "Circuito C", "muscles": [], "focus": "Circuito"},
        {"name": "Circuito D", "muscles": [], "focus": "Circuito"},
    ]

    def pick_exercises(muscles: List[str], obj: str) -> List[str]:
        """Seleciona uma lista de exercícios com base nos grupamentos ou cardio."""
        exercises: List[str] = []
        if obj in {"hipertrofia", "forca"}:
            for muscle in muscles:
                choices = exercise_dict.get(muscle, [])
                if choices:
                    # Seleciona até 2 exercícios aleatórios por grupamento para dar variedade
                    selected = random.sample(choices, min(2, len(choices)))
                    exercises.extend(selected)
            random.shuffle(exercises)
            # Limita a 6 exercícios no máximo
            exercises = exercises[:6]
        else:
            # Emagrecimento e condicionamento usam exercícios de cardio/funcionais
            exercises = random.sample(cardio_exercises, 5)
        return exercises

    # Loop para construir as semanas e sessões
    for week_num in range(1, weeks + 1):
        week_sessions: List[Dict[str, Any]] = []
        # Dentro de cada semana geramos `freq` sessões ou até atingir 12 no total
        for day in range(1, freq + 1):
            if sessions_generated >= total_sessions:
                break
            if goal == "hipertrofia":
                split = hypertrophy_splits[sessions_generated % len(hypertrophy_splits)]
            elif goal == "forca":
                split = strength_splits[sessions_generated % len(strength_splits)]
            elif goal == "condicionamento":
                split = conditioning_splits[sessions_generated % len(conditioning_splits)]
            else:  # emagrecimento
                split = emagrecimento_splits[sessions_generated % len(emagrecimento_splits)]
            exercises = pick_exercises(split["muscles"], goal)
            ex_list: List[Dict[str, Any]] = []
            for ex in exercises:
                if goal == "hipertrofia":
                    ex_list.append(
                        {
                            "exercise": ex,
                            "sets": 4,
                            "reps": "8-12",
                            "rest": "60-90s",
                            "notes": "",
                        }
                    )
                elif goal == "forca":
                    ex_list.append(
                        {
                            "exercise": ex,
                            "sets": 4,
                            "reps": "4-6",
                            "rest": "2-3min",
                            "notes": "",
                        }
                    )
                elif goal == "emagrecimento":
                    ex_list.append(
                        {
                            "exercise": ex,
                            "sets": 3,
                            "reps": "12-15",
                            "rest": "30s",
                            "notes": "",
                        }
                    )
                else:  # condicionamento
                    ex_list.append(
                        {
                            "exercise": ex,
                            "sets": 3,
                            "reps": "10-15",
                            "rest": "30-60s",
                            "notes": "",
                        }
                    )
            session = {
                "day": day,
                "name": split["name"],
                "focus": split["focus"],
                "resumo": "",
                "exercises": ex_list,
                "conditioning": None,
                "mobility": None,
                "progression": None,
            }
            week_sessions.append(session)
            sessions_generated += 1
        training_weeks.append(
            {
                "week": week_num,
                "focus": f"Semana de {goal}",
                "sessions": week_sessions,
            }
        )
        if sessions_generated >= total_sessions:
            break
    # Construção do dicionário final
    plan_payload: Dict[str, Any] = {
        "overview": {
            "primary_goal": goal,
            "focus_points": [goal],
            "macrocycle_length_weeks": weeks,
        },
        "guidelines": [
            "Realize um aquecimento de 5-10 minutos antes das sessões",
            "Mantenha-se hidratado durante todo o treino",
            "Foque na execução correta dos movimentos",
        ],
        "training_weeks": training_weeks,
        "recovery": [
            "Durma pelo menos 7-8 horas por noite",
            "Inclua alongamentos após os treinos",
            "Faça dias de descanso ativo conforme necessário",
        ],
        "nutrition_tips": [
            "Ajuste sua alimentação de acordo com o objetivo",
            "Inclua proteínas magras e carboidratos complexos",
            "Beba bastante água",
        ],
        "progression_strategy": "Aumente gradualmente a carga ou repetições a cada semana conforme se sentir confortável.",
    }
    return plan_payload


def render_plan_markdown(plan_payload: Dict[str, Any]) -> str:
    """Converte o dicionário de plano em texto Markdown legível.

    Esta função segue de perto a implementação do notebook para renderizar
    o plano final em Markdown, com seções organizadas e listas formatadas.
    """
    lines: List[str] = []
    overview = plan_payload.get("overview", {})
    if overview:
        lines.append("## Visão geral")
        primary_goal = overview.get("primary_goal")
        if primary_goal:
            lines.append(f"- **Objetivo principal:** {primary_goal}")
        focus_points = overview.get("focus_points")
        if focus_points:
            if isinstance(focus_points, list):
                focus_str = ", ".join(str(fp) for fp in focus_points)
            else:
                focus_str = str(focus_points)
            lines.append(f"- **Pontos de foco:** {focus_str}")
        macro_weeks = overview.get("macrocycle_length_weeks")
        if macro_weeks:
            lines.append(f"- **Duração do macrociclo:** {macro_weeks} semanas")
        lines.append("")
    guidelines = plan_payload.get("guidelines")
    if guidelines:
        lines.append("## Diretrizes")
        for item in guidelines:
            lines.append(f"- {item}")
        lines.append("")
    training_weeks = plan_payload.get("training_weeks")
    if training_weeks:
        lines.append("## Semanas de treino")
        for week in training_weeks:
            week_num = week.get("week")
            focus = week.get("focus")
            if week_num is not None:
                header = f"### Semana {week_num}"
                if focus:
                    header += f": {focus}"
                lines.append(header)
            sessions = week.get("sessions", [])
            for session_idx, session in enumerate(sessions, 1):
                day = session.get("day")
                name = session.get("name") or f"Sessão {session_idx}"
                focus_sess = session.get("focus")
                session_header = f"#### {name}" if name else "#### Sessão"
                if focus_sess:
                    session_header += f" – {focus_sess}"
                lines.append(session_header)
                resumo = session.get("resumo")
                if resumo:
                    lines.append(f"{resumo}")
                exercises = session.get("exercises", [])
                if exercises:
                    lines.append("##### Exercícios")
                    for ex in exercises:
                        exercise_name = ex.get("exercise")
                        sets = ex.get("sets")
                        reps = ex.get("reps")
                        rest = ex.get("rest")
                        notes = ex.get("notes")
                        parts: List[str] = []
                        if sets is not None:
                            parts.append(f"{sets}x")
                        if reps is not None:
                            parts.append(str(reps))
                        if rest:
                            parts.append(f"descanso {rest}")
                        rep_str = " ".join(parts)
                        line = f"- {exercise_name} {rep_str}" if exercise_name else "-"
                        if notes:
                            line += f" – {notes}"
                        lines.append(line)
                conditioning = session.get("conditioning")
                if conditioning:
                    lines.append(f"- **Condicionamento:** {conditioning}")
                mobility = session.get("mobility")
                if mobility:
                    lines.append(f"- **Mobilidade:** {mobility}")
                progression = session.get("progression")
                if progression:
                    lines.append(f"- **Progressão:** {progression}")
                lines.append("")
        lines.append("")
    recovery = plan_payload.get("recovery")
    if recovery:
        lines.append("## Recuperação")
        for item in recovery:
            lines.append(f"- {item}")
        lines.append("")
    nutrition = plan_payload.get("nutrition_tips")
    if nutrition:
        lines.append("## Dicas de nutrição")
        for tip in nutrition:
            lines.append(f"- {tip}")
        lines.append("")
    progression_strategy = plan_payload.get("progression_strategy")
    if progression_strategy:
        lines.append("## Estratégia de progressão")
        lines.append(progression_strategy)
        lines.append("")
    return "\n".join(lines)


def formatar_analises(analises: Dict[str, Any]) -> str:
    """Formata as métricas analíticas em uma lista de texto."""
    if not analises:
        return ""
    partes = [
        f"- Calorias estimadas por sessão: {analises.get('estimated_session_calories')} kcal",
        f"- Calorias semanais aproximadas: {analises.get('estimated_weekly_calories')} kcal",
        f"- Faixa de repetições sugerida: {analises.get('recommended_rep_range')}",
        f"- Intensidade sugerida: {analises.get('recommended_intensity')}",
        f"- Volume por sessão: {analises.get('volume_per_session')}",
    ]
    if analises.get('bmi'):
        partes.append(f"- IMC aproximado: {analises['bmi']}")
    return "\n".join(partes)


def resumir_pesquisa_web(pesquisa: List[Dict[str, Any]]) -> List[str]:
    """Gera uma lista de resumos a partir dos resultados de pesquisa.

    Para cada consulta, pega o primeiro resultado (título ou URL) e gera
    uma frase curta que pode ser apresentada ao usuário como referência.
    """
    summaries: List[str] = []
    for item in pesquisa or []:
        query = item.get("query", "")
        results = item.get("results", [])
        resumo: Optional[str] = None
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                title = first.get("title") or first.get("url") or first.get("snippet")
                resumo = title if title else str(first)
            else:
                resumo = str(first)
        if resumo:
            summaries.append(f"{query}: {resumo}")
        else:
            summaries.append(query)
    return summaries


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    st.set_page_config(page_title="AI Fitness Coach", layout="wide")
    st.title("🏋️ AI Fitness Coach – Plano de Musculação Personalizado")
    st.write(
        "Preencha os dados abaixo para gerar um plano de treino personalizado com 12 sessões."
    )
    # Colunas para organizar a interface
    col1, col2 = st.columns([1, 1])
    with col1:
        age = st.number_input("Idade", min_value=15, max_value=80, value=25, step=1)
        weight = st.number_input(
            "Peso corporal (kg)", min_value=35.0, max_value=250.0, value=75.0, step=0.5
        )
        height_input = st.text_input(
            "Altura (cm) (opcional)", placeholder="Ex: 175"
        )
        freq = st.selectbox("Frequência de treinos por semana", [2, 3, 4, 5, 6], index=2)
        primary_goal_display = st.selectbox(
            "Objetivo principal",
            [
                "Hipertrofia",
                "Emagrecimento",
                "Força",
                "Condicionamento",
            ],
            index=0,
        )
        experience_display = st.selectbox(
            "Nível de experiência",
            ["Iniciante", "Intermediário", "Avançado"],
            index=0,
        )
        restrictions = st.text_input(
            "Restrições (lesões, limitações) (opcional)", placeholder="Ex: dor no ombro"
        )
        notes = st.text_input(
            "Notas adicionais (opcional)", placeholder="Observações gerais"
        )
        generate = st.button("Gerar plano", type="primary")
    # Área de saída
    with col2:
        if generate:
            # Normalização de entradas
            user_profile: Dict[str, Any] = {
                "age": int(age),
                "weight_kg": float(weight),
                "training_frequency": int(freq),
                "primary_goal": normalizar_objetivo(primary_goal_display),
                "experience_level": normalizar_experiencia(experience_display),
                "restrictions": restrictions.strip() or None,
                "additional_notes": notes.strip() or None,
            }
            if height_input.strip():
                try:
                    user_profile["height_cm"] = float(height_input.strip())
                except ValueError:
                    st.error("Altura inválida. Informe somente números (ex: 175).")
                    return
            # Cálculos analíticos
            try:
                analises = ferramenta_matematica_treino(
                    age=user_profile["age"],
                    weight_kg=user_profile["weight_kg"],
                    training_frequency=user_profile["training_frequency"],
                    primary_goal=user_profile["primary_goal"],
                    height_cm=user_profile.get("height_cm"),
                    experience_level=user_profile.get("experience_level", "iniciante"),
                )
            except Exception as exc:
                st.error(f"Erro nos cálculos: {exc}")
                return
            # Pesquisas na web – três queries diferentes para enriquecer o plano
            queries = [
                f"treino de musculação {user_profile['primary_goal']} {user_profile['training_frequency']} dias por semana",
                f"melhores exercícios para {user_profile['primary_goal']} academia",
                f"estratégias de treino para {user_profile['primary_goal']}",
            ]
            research_payload: List[Dict[str, Any]] = []
            for q in queries:
                results = search_web(q, max_results=4)
                research_payload.append({"query": q, "results": results})
            # Gerar plano de treino
            plan_payload = generate_plan(user_profile, analises)
            # Converter plano em Markdown para exibição
            plan_markdown = render_plan_markdown(plan_payload)
            # Mostrar perfil validado
            st.subheader("Perfil validado")
            st.markdown(f"- **Idade:** {user_profile['age']} anos")
            st.markdown(f"- **Peso:** {user_profile['weight_kg']} kg")
            if user_profile.get("height_cm"):
                st.markdown(f"- **Altura:** {user_profile['height_cm']} cm")
            st.markdown(
                f"- **Frequência semanal:** {user_profile['training_frequency']} dias"
            )
            st.markdown(f"- **Objetivo:** {user_profile['primary_goal']}")
            st.markdown(f"- **Nível:** {user_profile['experience_level']}")
            # Mostrar análises
            st.subheader("Insights analíticos")
            st.markdown(formatar_analises(analises))
            # Mostrar resumo de pesquisa
            resumo = resumir_pesquisa_web(research_payload)
            if resumo:
                st.subheader("Referências encontradas na web")
                for item in resumo:
                    st.markdown(f"- {item}")
            # Mostrar plano final
            st.subheader("Plano final")
            st.markdown(plan_markdown)


if __name__ == "__main__":
    main()