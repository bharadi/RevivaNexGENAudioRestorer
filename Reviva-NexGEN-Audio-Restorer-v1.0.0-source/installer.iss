; Inno Setup Script for Reviva NexGEN Audio Restorer
#define MyAppName "Reviva NexGEN Audio Restorer"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Dr. Vinayak Ashok Bharadi"
#define MyAppContact "vinayagaurang@zohomail.in"
#define MyAppExeName "nexgen_desktop_rust.exe"

[Setup]
AppId={{7A4E91C2-5D3B-4A8F-9C21-6E8B42D1F904}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppContact={#MyAppContact}
DefaultDirName={autopf}\RevivaNexGENAudioRestorer
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=Output
OutputBaseFilename=Reviva-NexGEN-Audio-Restorer-Setup-v1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
SetupIconFile=assets\app_icon.ico
LicenseFile=TERMS_OF_USE.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Create a Quick Launch shortcut"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "TERMS_OF_USE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "models\deepfilternet3.onnx"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "models\IRC_1002_C.bin"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "models\README.md"; DestDir: "{app}\models"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
