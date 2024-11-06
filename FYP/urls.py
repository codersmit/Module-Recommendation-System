"""
URL configuration for FYP project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from OptiMods import views
from OptiMods.views import submit_feedback

urlpatterns = [
    path('', views.my_view, name='home'),
    path('register/', views.register, name='register'),
    path('registration_success/', views.registration_success, name='registration_success'),
    path('admin/', admin.site.urls),
    path('activate/<str:uidb64>/<str:token>/', views.activate_account, name='activate_account'),
    path('accounts/', include('allauth.urls')),
    path('login/', views.user_login, name='user_login'),
    path('login_success/', views.login_success, name='login_success'),
    path('logout-success/', views.logout_success, name='logout_success'),
    path('preferences/', views.preferences, name='preferences'),
    path('calculate_score/', views.calculate_score, name='calculate_score'),
    path('submit-feedback/', submit_feedback, name='submit_feedback'),
    path('feedback-success/', views.feedback_success, name='feedback_success'),
]
