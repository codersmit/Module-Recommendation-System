# Generated by Django 5.0.1 on 2024-02-20 14:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('OptiMods', '0002_alter_user_first_name_alter_user_groups_and_more'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='first_name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='user',
            name='groups',
            field=models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_groups', to='auth.group', verbose_name='Groups'),
        ),
        migrations.AlterField(
            model_name='user',
            name='last_name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='user',
            name='user_permissions',
            field=models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_permissions', to='auth.permission', verbose_name='User permissions'),
        ),
    ]
