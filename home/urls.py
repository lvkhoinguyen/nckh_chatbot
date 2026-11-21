from django.urls import path
from . import views


urlpatterns = [
    path("", views.chatGoD, name="home"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/chat/history/", views.api_chat_history, name="api_chat_history"),
    path("api/chat/clear/", views.api_chat_clear, name="api_chat_clear"),
    path("api/documents/select/", views.api_select_documents, name="api_select_documents"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("account/", views.account, name="account"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload, name="upload"),
    path("selected/", views.select_files, name="select_file"),
]
