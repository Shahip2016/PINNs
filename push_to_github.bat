@echo off
echo ============================================================
echo         PUSH PINNS PROJECT TO GITHUB
echo ============================================================
echo.

echo This script will help you push your PINNs project to GitHub.
echo.
echo PREREQUISITES:
echo 1. You must have a GitHub account
echo 2. You must create a new repository on GitHub first
echo.
pause

echo.
echo ============================================================
echo STEP 1: Enter your GitHub username
echo ============================================================
set /p username="Enter your GitHub username: "

echo.
echo ============================================================
echo STEP 2: Enter your repository name
echo ============================================================
echo (Suggested: physics-informed-neural-networks)
set /p reponame="Enter your repository name: "

echo.
echo ============================================================
echo STEP 3: Adding GitHub remote
echo ============================================================
echo Adding remote: https://github.com/%username%/%reponame%.git
git remote add origin https://github.com/%username%/%reponame%.git

echo.
echo Checking remote status...
git remote -v

echo.
echo ============================================================
echo STEP 4: Pushing to GitHub
echo ============================================================
echo Pushing to main branch...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo ALTERNATIVE: If push failed, trying with master branch
    echo ============================================================
    git push -u origin master
)

echo.
echo ============================================================
echo COMPLETED!
echo ============================================================
echo Your repository should now be available at:
echo https://github.com/%username%/%reponame%
echo.
echo Next steps:
echo 1. Visit your repository on GitHub
echo 2. Add topics: machine-learning, physics-informed-neural-networks, pde-solver
echo 3. Update the README if needed
echo 4. Star the repository to make it easier to find!
echo.
pause
