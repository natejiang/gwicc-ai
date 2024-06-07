import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from functools import wraps

import flask
import requests
from flask import Flask, Response, jsonify, request
from flask_sock import Sock

from .assets import css_content, html_content, js_content
from .auth import AuthInterface, NoAuth


class Cache(ABC):
    """
    ABC在Python中代表"Abstract Base Class"(抽象基类)
    Define the interface for a cache that can be used to store data in a Flask app.
    """

    @abstractmethod
    def generate_id(self, *args, **kwargs):
        """
        Generate a unique ID for the cache.
        """
        pass

    @abstractmethod
    def get(self, id, field):
        """
        Get a value from the cache.
        """
        pass

    @abstractmethod
    def get_all(self, field_list) -> list:
        """
        Get all values from the cache.
        """
        pass

    @abstractmethod
    def set(self, id, field, value):
        """
        Set a value in the cache.
        """
        pass

    @abstractmethod
    def delete(self, id):
        """
        Delete a value from the cache.
        """
        pass


class MemoryCache(Cache):
    def __init__(self):
        # 初始化一个空的字典来作为缓存，用于存储对象的相关数据
        self.cache = {}

    def generate_id(self, *args, **kwargs):
         # 使用uuid4生成一个随机的、全球唯一的标识符，并将其转换为字符串形式
        return str(uuid.uuid4())

    def set(self, id, field, value):
        # 检查缓存中是否存在指定的ID，如果不存在，则初始化一个空字典
        if id not in self.cache:
            self.cache[id] = {}
         # 将给定的字段-值对存储在缓存中指定的ID下
        self.cache[id][field] = value

    def get(self, id, field):
        if id not in self.cache:
            return None

        if field not in self.cache[id]:
            return None

        return self.cache[id][field]

    def get_all(self, field_list) -> list:
        return [
            {"id": id, **{field: self.get(id=id, field=field) for field in field_list}}
            for id in self.cache
        ]

    def delete(self, id):
        if id in self.cache:
            del self.cache[id]

class VannaFlaskApp:
    flask_app = None

    def requires_cache(self, required_fields, optional_fields=[]):
        """
        创建一个装饰器，确保在调用被装饰的函数之前，缓存中存在特定的字段。

        :param required_fields: 必需的字段名列表，这些字段必须存在于缓存中。
        :param optional_fields: 可选的字段名列表，这些字段如果存在则从缓存中获取。
        :return: 返回一个装饰器函数。
        """
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                # 尝试从查询参数中获取id
                id = request.args.get("id")

                # 如果查询参数中没有id，尝试从请求体中获取
                if id is None:
                    id = request.json.get("id")
                    # 如果仍然找不到id，返回错误信息
                    if id is None:
                        return jsonify({"type": "error", "error": "未提供id"})

                # 检查所有必需的字段是否都在缓存中
                for field in required_fields:
                    # 如果缺少任何一个必需字段，返回错误信息
                    if self.cache.get(id=id, field=field) is None:
                        return jsonify({"type": "error", "error": f"未找到 {field}"})

                # 从缓存中收集必需字段的值
                field_values = {
                    field: self.cache.get(id=id, field=field) for field in required_fields
                }

                # 添加可选字段的值（如果存在）到缓存中
                for field in optional_fields:
                    field_values[field] = self.cache.get(id=id, field=field)

                # 将id添加到字段值中
                field_values["id"] = id

                # 调用被装饰的函数，传入收集到的字段值
                return f(*args, **field_values, **kwargs)

            return decorated

        return decorator

    def requires_auth(self, f):
        """
        身份验证装饰器函数，用于在调用原函数前检查用户是否已认证。
        
        如用户未认证，返回包含登录表单的JSON响应。
        若用户已认证，将用户对象传递给原函数并调用它。
        
        :param f: 需要被包装的原函数
        :return: 包装后的函数
        """
        @wraps(f)
        def decorated(*args, **kwargs):
            # 根据当前请求从认证模块获取用户信息
            user = self.auth.get_user(flask.request)

            # 检查用户是否已登录
            if not self.auth.is_logged_in(user):
                # 若未登录，返回含有登录表单的JSON响应
                return jsonify({"type": "未登录", "html": self.auth.login_form()})

            # 若已登录，调用原函数并传入用户对象
            # 将用户信息传递给函数
            return f(*args, user=user, **kwargs)

        return decorated

    def __init__(self, vn, cache: Cache = MemoryCache(),
                    auth: AuthInterface = NoAuth(),
                    debug=True,
                    allow_llm_to_see_data=False,
                    logo="https://img.vanna.ai/vanna-flask.svg",
                    title="Welcome to Vanna.AI",
                    subtitle="Your AI-powered copilot for SQL queries.",
                    show_training_data=True,
                    suggested_questions=False,
                    sql=False,
                    table=True,
                    csv_download=True,
                    chart=False,
                    redraw_chart=False,
                    auto_fix_sql=True,
                    ask_results_correct=True,
                    followup_questions=False,
                    summarization=False
                 ):
        """
        Expose a Flask app that can be used to interact with a Vanna instance.

        Args:
            vn: The Vanna instance to interact with.
            cache: The cache to use. Defaults to MemoryCache, which uses an in-memory cache. You can also pass in a custom cache that implements the Cache interface.
            auth: The authentication method to use. Defaults to NoAuth, which doesn't require authentication. You can also pass in a custom authentication method that implements the AuthInterface interface.
            debug: Show the debug console. Defaults to True.
            allow_llm_to_see_data: Whether to allow the LLM to see data. Defaults to False.
            logo: The logo to display in the UI. Defaults to the Vanna logo.
            title: The title to display in the UI. Defaults to "Welcome to Vanna.AI".
            subtitle: The subtitle to display in the UI. Defaults to "Your AI-powered copilot for SQL queries.".
            show_training_data: Whether to show the training data in the UI. Defaults to True.
            suggested_questions: Whether to show suggested questions in the UI. Defaults to True.
            sql: Whether to show the SQL input in the UI. Defaults to True.
            table: Whether to show the table output in the UI. Defaults to True.
            csv_download: Whether to allow downloading the table output as a CSV file. Defaults to True.
            chart: Whether to show the chart output in the UI. Defaults to True.
            redraw_chart: Whether to allow redrawing the chart. Defaults to True.
            auto_fix_sql: Whether to allow auto-fixing SQL errors. Defaults to True.
            ask_results_correct: Whether to ask the user if the results are correct. Defaults to True.
            followup_questions: Whether to show followup questions. Defaults to True.
            summarization: Whether to show summarization. Defaults to True.

        Returns:
            None
        """
        self.flask_app = Flask(__name__)
        self.sock = Sock(self.flask_app)
        self.ws_clients = []
        self.vn = vn
        self.debug = debug
        self.auth = auth
        self.cache = cache
        self.allow_llm_to_see_data = allow_llm_to_see_data
        self.logo = logo
        self.title = title
        self.subtitle = subtitle
        self.show_training_data = show_training_data
        self.suggested_questions = suggested_questions
        self.sql = sql
        self.table = table
        self.csv_download = csv_download
        self.chart = chart
        self.redraw_chart = redraw_chart
        self.auto_fix_sql = auto_fix_sql
        self.ask_results_correct = ask_results_correct
        self.followup_questions = followup_questions
        self.summarization = summarization

        # 初始化日志系统，命名为"werkzeug"
        log = logging.getLogger("werkzeug")
        
        # 将日志级别设置为ERROR，只记录错误级别的日志信息
        log.setLevel(logging.ERROR)

        if "google.colab" in sys.modules:
            self.debug = False
            print("Google Colab doesn't support running websocket servers. Disabling debug mode.")

        if self.debug:
            def log(message, title="Info"):
                [ws.send(json.dumps({'message': message, 'title': title})) for ws in self.ws_clients]

            self.vn.log = log

        @self.flask_app.route("/auth/login", methods=["POST"])
        def login():
            """
            处理用户登录请求。

            本函数通过调用auth模块的login_handler方法，处理Flask框架接收到的登录请求。
            它不接受任何参数，直接使用flask.request对象作为login_handler的方法参数。

            返回值:
            - 根据auth模块的login_handler方法的实现，返回相应的处理结果，例如重定向到登录成功页面或返回错误信息。
            """
            return self.auth.login_handler(flask.request)

        @self.flask_app.route("/auth/callback", methods=["GET"])
        def callback():
            """
            处理认证回调。

            该函数用于处理外部服务（如OAuth提供商）的认证回调请求。它通过解析Flask请求对象，
            并将其传递给认证模块的回调处理器，来处理认证流程的最后一步。

            返回:
                处理回调请求后的响应对象。
            """
            return self.auth.callback_handler(flask.request)

        @self.flask_app.route("/auth/logout", methods=["GET"])
        def logout():
            return self.auth.logout_handler(flask.request)

        @self.flask_app.route("/api/v0/get_config", methods=["GET"])
        @self.requires_auth
        def get_config(user: any):
            """
            根据用户信息获取配置信息。
            
            此函数生成一个包含各种配置设置的字典，这些设置可能影响应用程序的行为。
            这些配置设置包括调试模式、徽标、标题、副标题、显示训练数据等。
            接着，根据用户的权限，使用auth模块的override_config_for_user方法来覆盖或修改这些配置。
            最后，将配置信息封装为JSON格式，以便通过API返回给客户端。
            
            参数:
            user: any - 当前请求的用户信息。用户信息可以是任何类型，具体取决于应用程序的实现。
            
            返回:
            jsonify - 包含配置信息的JSON响应对象。
            """
            # 初始化配置字典，包含各种应用程序配置设置
            config = {
                "debug": self.debug,
                "logo": self.logo,
                "title": self.title,
                "subtitle": self.subtitle,
                "show_training_data": self.show_training_data,
                "suggested_questions": self.suggested_questions,
                "sql": self.sql,
                "table": self.table,
                "csv_download": self.csv_download,
                "chart": self.chart,
                "redraw_chart": self.redraw_chart,
                "auto_fix_sql": self.auto_fix_sql,
                "ask_results_correct": self.ask_results_correct,
                "followup_questions": self.followup_questions,
                "summarization": self.summarization,
            }

            # 根据用户权限覆盖或修改配置
            config = self.auth.override_config_for_user(user, config)

            # 将配置信息封装为JSON格式，准备返回给客户端
            return jsonify(
                {
                    "type": "config",
                    "config": config
                }
            )

        @self.flask_app.route("/api/v0/generate_questions", methods=["GET"])
        @self.requires_auth
        def generate_questions(user: any):
            # If self has an _model attribute and model=='chinook'
            if hasattr(self.vn, "_model") and self.vn._model == "chinook":
                return jsonify(
                    {
                        "type": "question_list",
                        "questions": [
                            "What are the top 10 artists by sales?",
                            "What are the total sales per year by country?",
                            "Who is the top selling artist in each genre? Show the sales numbers.",
                            "How do the employees rank in terms of sales performance?",
                            "Which 5 cities have the most customers?",
                        ],
                        "header": "Here are some questions you can ask:",
                    }
                )

            training_data = vn.get_training_data()

            # If training data is None or empty
            if training_data is None or len(training_data) == 0:
                return jsonify(
                    {
                        "type": "error",
                        "error": "No training data found. Please add some training data first.",
                    }
                )

            # Get the questions from the training data
            try:
                # Filter training data to only include questions where the question is not null
                questions = (
                    training_data[training_data["question"].notnull()]
                    .sample(5)["question"]
                    .tolist()
                )

                # Temporarily this will just return an empty list
                return jsonify(
                    {
                        "type": "question_list",
                        "questions": questions,
                        "header": "Here are some questions you can ask",
                    }
                )
            except Exception as e:
                return jsonify(
                    {
                        "type": "question_list",
                        "questions": [],
                        "header": "Go ahead and ask a question",
                    }
                )

        @self.flask_app.route("/api/v0/generate_sql", methods=["GET"])
        @self.requires_auth
        def generate_sql(user: any):
            """
            根据用户请求生成SQL查询语句。

            该函数处理来自Flask应用程序的HTTP请求，并根据请求中的“question”参数生成相应的SQL语句。
            如果生成的SQL语句有效，则返回包含SQL信息的JSON响应；否则，返回包含错误信息的JSON响应。

            :param user: 当前操作的用户信息，参数类型为任意类型。
            :return: 根据情况返回包含SQL信息或错误信息的JSON响应。
            """
            # 从请求参数中获取问题
            question = flask.request.args.get("question")

            # 检查是否提供了问题，如果没有提供，则返回错误信息
            if question is None:
                return jsonify({"type": "error", "error": "No question provided"})

            # 为问题生成唯一ID，并根据问题生成SQL语句
            id = self.cache.generate_id(question=question)
            sql = vn.generate_sql(question=question, allow_llm_to_see_data=self.allow_llm_to_see_data)

            # 将问题和生成的SQL语句存储到缓存中
            self.cache.set(id=id, field="question", value=question)
            self.cache.set(id=id, field="sql", value=sql)

            # 检查生成的SQL语句是否有效，根据结果返回不同的JSON响应
            if vn.is_sql_valid(sql=sql):
                return jsonify(
                    {
                        "type": "sql",
                        "id": id,
                        "text": sql,
                    }
                )
            else:
                return jsonify(
                    {
                        "type": "text",
                        "id": id,
                        "text": sql,
                    }
                )

        @self.flask_app.route("/api/v0/run_sql", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["sql"])
        def run_sql(user: any, id: str, sql: str):
            try:
                if not vn.run_sql_is_set:
                    return jsonify(
                        {
                            "type": "error",
                            "error": "Please connect to a database using vn.connect_to_... in order to run SQL queries.",
                        }
                    )
                
                df = vn.run_sql(sql=sql)

                self.cache.set(id=id, field="df", value=df)

                return jsonify(
                    {
                        "type": "df",
                        "id": id,
                        "df": df.head(10).to_json(orient='records', date_format='iso'),
                        "should_generate_chart": self.chart and vn.should_generate_chart(df),
                    }
                )

            except Exception as e:
                return jsonify({"type": "sql_error", "error": str(e)})

        @self.flask_app.route("/api/v0/fix_sql", methods=["POST"])
        @self.requires_auth
        @self.requires_cache(["question", "sql"])
        def fix_sql(user: any, id: str, question:str, sql: str):
            error = flask.request.json.get("error")

            if error is None:
                return jsonify({"type": "error", "error": "No error provided"})

            question = f"I have an error: {error}\n\nHere is the SQL I tried to run: {sql}\n\nThis is the question I was trying to answer: {question}\n\nCan you rewrite the SQL to fix the error?"

            fixed_sql = vn.generate_sql(question=question)

            self.cache.set(id=id, field="sql", value=fixed_sql)

            return jsonify(
                {
                    "type": "sql",
                    "id": id,
                    "text": fixed_sql,
                }
            )


        @self.flask_app.route('/api/v0/update_sql', methods=['POST'])
        @self.requires_auth
        @self.requires_cache([])
        def update_sql(user: any, id: str):
            sql = flask.request.json.get('sql')

            if sql is None:
                return jsonify({"type": "error", "error": "No sql provided"})

            self.cache.set(id=id, field='sql', value=sql)

            return jsonify(
                {
                    "type": "sql",
                    "id": id,
                    "text": sql,
                })

        @self.flask_app.route("/api/v0/download_csv", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df"])
        def download_csv(user: any, id: str, df):
            csv = df.to_csv()

            return Response(
                csv,
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={id}.csv"},
            )

        @self.flask_app.route("/api/v0/generate_plotly_figure", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question", "sql"])
        def generate_plotly_figure(user: any, id: str, df, question, sql):
            chart_instructions = flask.request.args.get('chart_instructions')

            if chart_instructions is not None:
                question = f"{question}. When generating the chart, use these special instructions: {chart_instructions}"

            try:
                code = vn.generate_plotly_code(
                    question=question,
                    sql=sql,
                    df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                )
                fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
                fig_json = fig.to_json()

                self.cache.set(id=id, field="fig_json", value=fig_json)

                return jsonify(
                    {
                        "type": "plotly_figure",
                        "id": id,
                        "fig": fig_json,
                    }
                )
            except Exception as e:
                # Print the stack trace
                import traceback

                traceback.print_exc()

                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/get_training_data", methods=["GET"])
        @self.requires_auth
        def get_training_data(user: any):
            df = vn.get_training_data()

            if df is None or len(df) == 0:
                return jsonify(
                    {
                        "type": "error",
                        "error": "No training data found. Please add some training data first.",
                    }
                )

            return jsonify(
                {
                    "type": "df",
                    "id": "training_data",
                    "df": df.to_json(orient="records"),
                }
            )

        @self.flask_app.route("/api/v0/remove_training_data", methods=["POST"])
        @self.requires_auth
        def remove_training_data(user: any):
            # Get id from the JSON body
            id = flask.request.json.get("id")

            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})

            if vn.remove_training_data(id=id):
                return jsonify({"success": True})
            else:
                return jsonify(
                    {"type": "error", "error": "Couldn't remove training data"}
                )

        @self.flask_app.route("/api/v0/train", methods=["POST"])
        @self.requires_auth
        def add_training_data(user: any):
            question = flask.request.json.get("question")
            sql = flask.request.json.get("sql")
            ddl = flask.request.json.get("ddl")
            documentation = flask.request.json.get("documentation")

            try:
                id = vn.train(
                    question=question, sql=sql, ddl=ddl, documentation=documentation
                )

                return jsonify({"id": id})
            except Exception as e:
                print("TRAINING ERROR", e)
                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/generate_followup_questions", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question", "sql"])
        def generate_followup_questions(user: any, id: str, df, question, sql):
            if self.allow_llm_to_see_data:
                followup_questions = vn.generate_followup_questions(
                    question=question, sql=sql, df=df
                )
                if followup_questions is not None and len(followup_questions) > 5:
                    followup_questions = followup_questions[:5]

                self.cache.set(id=id, field="followup_questions", value=followup_questions)

                return jsonify(
                    {
                        "type": "question_list",
                        "id": id,
                        "questions": followup_questions,
                        "header": "Here are some potential followup questions:",
                    }
                )
            else:
                self.cache.set(id=id, field="followup_questions", value=[])
                return jsonify(
                    {
                        "type": "question_list",
                        "id": id,
                        "questions": [],
                        "header": "Followup Questions can be enabled if you set allow_llm_to_see_data=True",
                    }
                )

        @self.flask_app.route("/api/v0/generate_summary", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question"])
        def generate_summary(user: any, id: str, df, question):
            if self.allow_llm_to_see_data:
                summary = vn.generate_summary(question=question, df=df)

                self.cache.set(id=id, field="summary", value=summary)

                return jsonify(
                    {
                        "type": "text",
                        "id": id,
                        "text": summary,
                    }
                )
            else:
                return jsonify(
                    {
                        "type": "text",
                        "id": id,
                        "text": "Summarization can be enabled if you set allow_llm_to_see_data=True",
                    }
                )

        @self.flask_app.route("/api/v0/load_question", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(
            ["question", "sql", "df", "fig_json"],
            optional_fields=["summary"]
        )
        def load_question(user: any, id: str, question, sql, df, fig_json, summary):
            try:
                return jsonify(
                    {
                        "type": "question_cache",
                        "id": id,
                        "question": question,
                        "sql": sql,
                        "df": df.head(10).to_json(orient="records", date_format="iso"),
                        "fig": fig_json,
                        "summary": summary,
                    }
                )

            except Exception as e:
                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/get_question_history", methods=["GET"])
        @self.requires_auth
        def get_question_history(user: any):
            return jsonify(
                {
                    "type": "question_history",
                    "questions": cache.get_all(field_list=["question"]),
                }
            )

        @self.flask_app.route("/api/v0/<path:catch_all>", methods=["GET", "POST"])
        def catch_all(catch_all):
            return jsonify(
                {"type": "error", "error": "The rest of the API is not ported yet."}
            )

        @self.flask_app.route("/assets/<path:filename>")
        def proxy_assets(filename):
            if ".css" in filename:
                return Response(css_content, mimetype="text/css")

            if ".js" in filename:
                return Response(js_content, mimetype="text/javascript")

            # Return 404
            return "File not found", 404

        # Proxy the /vanna.svg file to the remote server
        @self.flask_app.route("/vanna.svg")
        def proxy_vanna_svg():
            remote_url = "https://vanna.ai/img/vanna.svg"
            response = requests.get(remote_url, stream=True)

            # Check if the request to the remote URL was successful
            if response.status_code == 200:
                excluded_headers = [
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "connection",
                ]
                headers = [
                    (name, value)
                    for (name, value) in response.raw.headers.items()
                    if name.lower() not in excluded_headers
                ]
                return Response(response.content, response.status_code, headers)
            else:
                return "Error fetching file from remote server", response.status_code

        if self.debug:
            @self.sock.route("/api/v0/log")
            def sock_log(ws):
                self.ws_clients.append(ws)

                try:
                    while True:
                        message = ws.receive()  # This example just reads and ignores to keep the socket open
                finally:
                    self.ws_clients.remove(ws)


        @self.flask_app.route("/", defaults={"path": ""})
        @self.flask_app.route("/<path:path>")
        def hello(path: str):
            return html_content

    def run(self, *args, **kwargs):
        """
        Run the Flask app.

        Args:
            *args: Arguments to pass to Flask's run method.
            **kwargs: Keyword arguments to pass to Flask's run method.

        Returns:
            None
        """
        if args or kwargs:
            self.flask_app.run(*args, **kwargs)

        else:
            try:
                from google.colab import output

                output.serve_kernel_port_as_window(8084)
                from google.colab.output import eval_js

                print("Your app is running at:")
                print(eval_js("google.colab.kernel.proxyPort(8084)"))
            except:
                print("Your app is running at:")
                print("http://localhost:8084")

            self.flask_app.run(host="0.0.0.0", port=8084, debug=self.debug)
