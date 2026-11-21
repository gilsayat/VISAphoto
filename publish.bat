@echo off
echo Packaging US Visa Photo Automator by Gil...
dotnet publish -c Release -r win-x64 /p:PublishSingleFile=true /p:SelfContained=true /p:IncludeNativeLibrariesForSelfExtract=true -o ./Publish
echo.
echo Build Complete!
echo IMPORTANT: Copy your "Models" folder into the new "Publish" directory so the AI has its required data files.
pause