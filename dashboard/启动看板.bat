@echo off
title Alpha191 因子看板
cd /d "%~dp0"
echo 正在打开 Alpha191 因子看板，请稍候...
echo （看板将在你的默认浏览器中打开，关闭浏览器标签页即可退出）
start "" "index.html"
timeout /t 3 >nul
exit
