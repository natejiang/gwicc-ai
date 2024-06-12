from flask import render_template, redirect, request, url_for, flash
from flask_login import login_user, logout_user, login_required, \
    current_user
from . import auth
from .. import db
from ..models import User
from ..email import send_email
from .forms import LoginForm, RegistrationForm

# 注册auth/login
@auth.route('/login', methods=['GET', 'POST'])
# 定义登录函数，用于处理用户登录逻辑
def login():
    # 创建LoginForm实例，用于处理表单数据
    form = LoginForm()
    
    # 检查表单数据是否通过验证
    if form.validate_on_submit():
        # 根据表单提交的邮箱（转为小写）查询用户信息
        user = User.query.filter_by(email=form.email.data.lower()).first()
        
        # 检查用户是否存在且密码正确
        if user is not None and user.verify_password(form.password.data):
            # 登录用户，并根据用户选择是否记住登录状态
            login_user(user, form.remember_me.data)
            
            # 获取并处理登录后的跳转地址
            next = request.args.get('next')
            if next is None or not next.startswith('/'):
                # next = url_for('main.index')
                next = '/'
            
            # 跳转到登录后的页面
            return redirect(next)
        
        # 登录失败，显示错误提示
        flash('Invalid email or password.')
    
    # 渲染登录页面模板，并传入登录表单实例
    return render_template('auth/login.html', form=form)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('main.index'))


@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data.lower(),
                    username=form.username.data,
                    password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('You can now login.')
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', form=form)
