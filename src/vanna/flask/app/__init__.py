from flask import Flask
from flask_bootstrap import Bootstrap
from flask_mail import Mail
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from ..config import config

bootstrap = Bootstrap()
mail = Mail()
moment = Moment()
db = SQLAlchemy()

login_manager = LoginManager()
login_manager.login_view = 'auth.login'


# 创建并配置应用程序实例
def create_app(config_name):
    # 初始化Flask应用程序实例
    app = Flask(__name__)
    # 从指定的配置对象加载配置
    app.config.from_object(config[config_name])
    # 初始化配置对象，并应用到应用程序
    config[config_name].init_app(app)

    # 初始化Flask-Bootstrap扩展
    bootstrap.init_app(app)
    # 初始化Flask-Mail扩展
    mail.init_app(app)
    # 初始化Flask-Moment扩展，用于日期时间的格式化
    moment.init_app(app)
    # 初始化Flask-SQLAlchemy数据库扩展
    db.init_app(app)
    # 初始化Flask-Login登录管理器
    login_manager.init_app(app)

    # 注册主应用程序的蓝图
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # 注册认证模块的蓝图，并设置URL前缀
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    # 返回配置好的应用程序实例
    return app
