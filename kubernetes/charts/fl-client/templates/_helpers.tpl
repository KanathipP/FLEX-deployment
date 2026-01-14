{{/*
Common labels
*/}}
{{- define "fl-client.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Client name (release name)
*/}}
{{- define "fl-client.name" -}}
{{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Client ID (release name + random suffix for uniqueness)
*/}}
{{- define "fl-client.clientId" -}}
{{ .Release.Name }}-{{ randAlphaNum 4 | lower }}
{{- end }}
