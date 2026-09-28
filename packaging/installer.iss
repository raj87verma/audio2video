; Inno Setup script for Audio2Video
;
; Builds a single Setup.exe that installs the PyInstaller-frozen app
; (produced by `pyinstaller packaging/audio2video.spec`, expected at
; dist/audio2video/ before this script runs) into a user-chosen directory,
; creates Start Menu / optional Desktop shortcuts, and registers a normal
; Windows uninstaller entry.
;
; Compiled by the GitHub Actions workflow via ISCC.exe (Inno Setup 6,
; pre-installed on windows-latest runners). Can also be compiled locally on
; Windows after installing Inno Setup (https://jrsoftware.org/isinfo.php):
;   ISCC.exe packaging\installer.iss
;
; Version is injected at build time via /DAppVersion=x.y.z on the ISCC
; command line; falls back to 0.0.0-dev for ad-hoc local builds so the
; script is still valid stand-alone.
#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

#define AppName "Audio2Video"
#define AppPublisher "Audio2Video Project"
#define AppURL "https://github.com/raj87verma/audio2video"
#define AppExeName "Audio2Video.exe"
; Path to the PyInstaller onedir output, relative to this .iss file.
#define DistDir "..\dist\audio2video"

[Setup]
AppId={{6E2C9C2C-2E9C-4E36-8C67-6E9C0F6B2A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
; Default install location is the user's own Local AppData, NOT
; Program Files under C:\ — this keeps the whole install off the C: system
; drive tree by default when the user's profile itself lives elsewhere,
; and — more importantly — avoids requiring Administrator privileges just
; to install the app. The install directory remains fully user-editable
; on the wizard's "Select Destination Location" page either way.
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=Audio2Video-Setup
SetupIconFile=app_icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Let the user pick any drive/folder they want during setup.
AllowNoIcons=yes
DirExistsWarning=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Pull in the entire PyInstaller onedir output (the frozen app, its
; bundled Python runtime, and all collected dependencies).
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove any cache/log files the app itself may have written next to the
; executable (it normally writes user data under %LOCALAPPDATA%\.audio2video
; instead, but this is a defensive cleanup in case of stray temp files).
Type: filesandordirs; Name: "{app}\__pycache__"
