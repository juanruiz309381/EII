# Script para Video Explicativo
## Parcial 5: Evaluación de Modelos LLM con Ollama

**Duración estimada:** 8-10 minutos
**Formato:** Screencast + presentación de resultados

---

## Estructura del Video

### Sección 1: Introducción (1 min)
### Sección 2: Demostración de Instalación (1.5 min)
### Sección 3: Presentación de Modelos (1 min)
### Sección 4: Ejecución de Pruebas (2 min)
### Sección 5: Análisis de Resultados (2 min)
### Sección 6: Conversación con Mejor Modelo (1.5 min)
### Sección 7: Conclusiones y Cierre (1 min)

---

## SECCIÓN 1: INTRODUCCIÓN (1 minuto)

### [Pantalla: Título + Tu foto/avatar]

**NARRACIÓN:**

> "Hola, soy [Tu Nombre], estudiante de Ingeniería de Información en el ITM. En este video voy a presentar mi proyecto del Parcial 5, donde implementé y evalué tres modelos de lenguaje de gran tamaño usando Ollama.

> Los objetivos de este proyecto fueron: implementar al menos 3 modelos LLM, realizar pruebas conversacionales sistemáticas, generar un informe de desempeño y establecer una conversación coherente con el modelo de mejor rendimiento.

> Los modelos que evaluaré son Llama 3.2 con 1 billón de parámetros, DeepSeek R1 con 1.5 billones, y Qwen 2.5 con 500 millones. Todos son modelos livianos que pueden ejecutarse en hardware convencional gracias a Ollama."

### [Transición: Animación de flecha hacia terminal]

---

## SECCIÓN 2: DEMOSTRACIÓN DE INSTALACIÓN (1.5 minutos)

### [Pantalla: Terminal en Linux/macOS]

**NARRACIÓN:**

> "Primero, veamos cómo instalar Ollama. Es sorprendentemente simple."

### [Mostrar comando]
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**NARRACIÓN:**

> "Con un solo comando descargamos e instalamos Ollama. Esto toma menos de un minuto."

### [Mostrar instalación completándose]

**NARRACIÓN:**

> "Una vez instalado, verificamos la versión..."

### [Mostrar comando]
```bash
ollama --version
```

**NARRACIÓN:**

> "Y ahora descargamos nuestros tres modelos. Esto puede tomar entre 5 y 15 minutos dependiendo de tu conexión a internet."

### [Mostrar comandos acelerados]
```bash
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:0.5b
```

### [Mostrar descarga en progreso - acelerar con edición]

**NARRACIÓN:**

> "Para verificar que los modelos están listos, ejecutamos..."

### [Mostrar comando]
```bash
ollama list
```

**NARRACIÓN:**

> "Perfecto. Tenemos nuestros tres modelos instalados y listos para probar."

---

## SECCIÓN 3: PRESENTACIÓN DE MODELOS (1 minuto)

### [Pantalla: Tabla comparativa visual]

**NARRACIÓN:**

> "Antes de las pruebas, conozcamos brevemente cada modelo."

### [Mostrar gráfico con logos/características]

**Llama 3.2 (1B):**
> "Llama 3.2 es desarrollado por Meta AI. Con 1 billón de parámetros, está optimizado para balance entre velocidad y calidad. Fue entrenado con 15 trillones de tokens en múltiples idiomas."

**DeepSeek R1 (1.5B):**
> "DeepSeek R1 de DeepSeek AI tiene 1.5 billones de parámetros y una arquitectura especializada en razonamiento lógico y matemático. Incluye mecanismos de 'cadenas de pensamiento'."

**Qwen 2.5 (0.5B):**
> "Qwen 2.5 de Alibaba es el más pequeño con 500 millones de parámetros. Diseñado para ultra-eficiencia, ideal para dispositivos con recursos limitados."

**NARRACIÓN:**

> "La pregunta clave es: ¿cuál se desempeña mejor en la práctica? Vamos a descubrirlo."

---

## SECCIÓN 4: EJECUCIÓN DE PRUEBAS (2 minutos)

### [Pantalla: Open WebUI o Terminal]

**NARRACIÓN:**

> "Para las pruebas, diseñé 8 categorías: razonamiento lógico, matemáticas, comprensión lectora, creatividad, conocimiento general, programación, análisis y conversación cotidiana."

### [Mostrar notebook Jupyter abierto]

**NARRACIÓN:**

> "Desarrollé un notebook de Jupyter con todas las pruebas automatizadas. Veamos cómo funciona."

### [Ejecutar celda de prueba]

**NARRACIÓN:**

> "Aquí envío la misma pregunta a los tres modelos y capturo la respuesta y el tiempo."

### [Mostrar ejemplo: "Escribe una función Python que calcule factorial"]

**NARRACIÓN:**

> "Por ejemplo, pidamos que escriban una función para calcular el factorial de un número."

### [Mostrar respuesta de Llama 3.2 - código en pantalla]

**NARRACIÓN:**

> "Llama 3.2 responde en 5.9 segundos con código correcto y bien documentado."

### [Mostrar respuesta de DeepSeek - código con comentarios extensos]

**NARRACIÓN:**

> "DeepSeek toma 7.4 segundos pero incluye comentarios muy detallados explicando el razonamiento."

### [Mostrar respuesta de Qwen - código minimalista]

**NARRACIÓN:**

> "Qwen responde en apenas 2.1 segundos con código funcional pero más minimalista."

**NARRACIÓN:**

> "Este patrón se repite: Qwen es el más rápido, DeepSeek el más detallado, y Llama balancea ambos."

### [Mostrar barra de progreso de las 8 categorías completándose - acelerar]

---

## SECCIÓN 5: ANÁLISIS DE RESULTADOS (2 minutos)

### [Pantalla: Gráficos de resultados]

**NARRACIÓN:**

> "Después de ejecutar todas las pruebas, estos son los resultados."

### [Mostrar gráfico de barras: Tiempo promedio de respuesta]

**NARRACIÓN:**

> "En tiempo de respuesta, Qwen lidera con 1.82 segundos promedio, Llama en el medio con 4.52, y DeepSeek es el más lento con 5.89 segundos."

### [Mostrar gráfico: Relevancia de respuestas]

**NARRACIÓN:**

> "Sin embargo, en relevancia de respuestas, DeepSeek lidera con 78.3%, seguido de Llama con 72.5%, y Qwen con 61.2%."

### [Mostrar tabla de tasa de éxito]

**NARRACIÓN:**

> "En confiabilidad, tanto Llama como Qwen lograron 100% de éxito. DeepSeek experimentó un timeout en una prueba matemática compleja."

### [Mostrar gráfico comparativo por categoría]

**NARRACIÓN:**

> "Analizando por categorías, vemos que DeepSeek destaca en razonamiento lógico y matemáticas, mientras que Qwen brilla en conversación cotidiana. Llama mantiene desempeño consistente en todas las categorías."

### [Mostrar scorecard final]

**NARRACIÓN:**

> "Aplicando un sistema de scoring ponderado considerando velocidad, calidad y confiabilidad, Llama 3.2 emerge como el ganador con 87.3 puntos, seguido de DeepSeek con 82.6 y Qwen con 76.8."

---

## SECCIÓN 6: CONVERSACIÓN CON MEJOR MODELO (1.5 minutos)

### [Pantalla: Open WebUI o terminal con Llama 3.2]

**NARRACIÓN:**

> "Como Llama 3.2 obtuvo el mejor desempeño general, establecí una conversación extendida de 5 turnos sobre inteligencia artificial."

### [Mostrar conversación - scroll suave]

**Usuario:** "Hola, ¿cómo estás? Me gustaría hablar sobre inteligencia artificial."

**Llama 3.2:** [Mostrar respuesta en pantalla]

**NARRACIÓN:**

> "El modelo mantiene un tono conversacional apropiado y sugiere temas específicos."

### [Siguiente turno - acelerar scroll]

**Usuario:** "¿Cuáles crees que son las aplicaciones más importantes de la IA en la actualidad?"

**Llama 3.2:** [Mostrar lista de aplicaciones]

**NARRACIÓN:**

> "Proporciona una lista estructurada de aplicaciones reales: salud, transporte, finanzas, agricultura."

### [Siguiente turno - acelerar]

**Usuario:** "¿Y cuáles son los principales desafíos éticos que enfrenta la IA?"

**Llama 3.2:** [Mostrar respuesta con desafíos éticos]

**NARRACIÓN:**

> "La respuesta es sofisticada, mencionando sesgos, privacidad, transparencia y desplazamiento laboral. Mantiene coherencia con el tema previo."

### [Últimos dos turnos - montage rápido]

**NARRACIÓN:**

> "En los turnos finales, discutimos cómo desarrollar IA responsablemente y el modelo cierra con recomendaciones prácticas. La coherencia, profundidad y naturalidad fueron excelentes en toda la conversación."

---

## SECCIÓN 7: CONCLUSIONES Y CIERRE (1 minuto)

### [Pantalla: Resumen visual con bullet points]

**NARRACIÓN:**

> "Para concluir, estos son los hallazgos principales de mi evaluación:"

### [Bullet point 1 aparece]

> "Primero: Llama 3.2 es el modelo más versátil, ofreciendo el mejor balance entre velocidad, calidad y confiabilidad."

### [Bullet point 2 aparece]

> "Segundo: DeepSeek R1 se especializa en tareas de razonamiento complejo, ideal si la velocidad no es crítica."

### [Bullet point 3 aparece]

> "Tercero: Qwen 2.5 prioriza eficiencia extrema, perfecto para prototipado rápido o dispositivos con recursos limitados."

### [Bullet point 4 aparece]

> "Cuarto: No existe un 'mejor modelo absoluto', sino modelos apropiados para contextos específicos."

### [Bullet point 5 aparece]

> "Y quinto: Ollama democratiza el acceso a modelos LLM sofisticados, permitiendo experimentación local sin costos operativos."

### [Pantalla: Llamado a la acción]

**NARRACIÓN:**

> "Este proyecto demuestra que la frontera de la IA es cada vez más accesible. Cualquiera con una laptop puede ahora experimentar con modelos que hace dos años requerían infraestructura de millones de dólares."

### [Pantalla: Recursos y links]

**NARRACIÓN:**

> "Todos los notebooks, scripts y el informe técnico completo están disponibles en el repositorio que aparece en pantalla. También incluí una guía paso a paso para que puedan replicar estos experimentos."

### [Pantalla: Agradecimientos]

**NARRACIÓN:**

> "Agradezco al profesor [Nombre] por la guía en este proyecto, y a la comunidad open source de Ollama, Meta, DeepSeek y Alibaba por hacer posible este trabajo."

### [Pantalla: Cierre con tu nombre y contacto]

**NARRACIÓN:**

> "Gracias por ver este video. Si tienen preguntas o comentarios, pueden contactarme en [tu email/github]. ¡Hasta pronto!"

### [Música de cierre - fade out]

---

## TIPS PARA LA GRABACIÓN

### Antes de Grabar:

1. **Preparar Entorno Visual**
   - Terminal con fuente grande (16-18pt)
   - Tema de alto contraste
   - Cerrar pestañas/aplicaciones innecesarias
   - Configurar resolución 1920x1080

2. **Prueba de Audio**
   - Usar micrófono externo si es posible
   - Eliminar ruido de fondo
   - Grabar 30 segundos de prueba
   - Ajustar niveles

3. **Script Impreso**
   - Imprimir este script
   - Marcar pausas y énfasis
   - Practicar 2-3 veces

4. **Assets Listos**
   - Gráficos exportados como PNG
   - Screenshots en carpeta organizada
   - Videos de ejecución pre-grabados

### Durante la Grabación:

1. **Secciones Independientes**
   - Grabar cada sección por separado
   - Permite rehacer segmentos sin perder todo
   - Facilita edición posterior

2. **Ritmo y Claridad**
   - Hablar 10-15% más lento de lo normal
   - Pausas de 1-2 segundos entre ideas
   - Enfatizar números y conclusiones clave

3. **Transiciones**
   - Pausa de 3 segundos entre secciones
   - Facilita cortes en edición
   - Espacio para agregar efectos

### Software Recomendado:

**Para Grabar:**
- **Linux:** OBS Studio (gratuito)
- **macOS:** QuickTime + iMovie
- **Windows:** OBS Studio + DaVinci Resolve (gratuito)

**Para Editar:**
- **Básico:** iMovie, Windows Video Editor
- **Avanzado:** DaVinci Resolve (gratuito, profesional)
- **Alternativa:** Kdenlive (Linux, gratuito)

**Para Subtítulos:**
- YouTube Auto-Captions (gratuito, luego corregir)
- Subtitle Edit (gratuito, Windows)
- Aegisub (gratuito, multiplataforma)

### Post-Producción:

1. **Edición de Audio**
   - Normalizar volumen
   - Eliminar clics y respiraciones fuertes
   - Añadir música de fondo sutil (10-15% volumen)

2. **Edición de Video**
   - Cortar pausas largas
   - Acelerar ejecuciones repetitivas (2x-4x)
   - Añadir zoom para destacar elementos pequeños
   - Transiciones suaves entre secciones

3. **Elementos Visuales**
   - Título y créditos profesionales
   - Lower thirds con tu nombre
   - Callouts para destacar información clave
   - Flechas/círculos para guiar atención

4. **Música (libre de derechos)**
   - YouTube Audio Library
   - FreeMusicArchive.org
   - Bensound.com
   - Incompetech.com

### Checklist Final:

- [ ] Audio claro sin distorsión
- [ ] Video en 1080p mínimo
- [ ] Subtítulos incluidos (español)
- [ ] Duración entre 8-12 minutos
- [ ] Todos los gráficos son legibles
- [ ] Código en pantalla es visible
- [ ] Transiciones suaves
- [ ] Música de fondo balanceada
- [ ] Créditos y referencias incluidos
- [ ] Exportado en formato compatible (MP4 H.264)

---

## STORYBOARD VISUAL

### Frame 1: Intro
```
┌────────────────────────┐
│                        │
│   EVALUACIÓN DE        │
│   MODELOS LLM          │
│   CON OLLAMA           │
│                        │
│   [Tu Nombre]          │
│   ITM 2025-2           │
└────────────────────────┘
```

### Frame 2: Terminal con Ollama
```
┌────────────────────────┐
│ $ ollama pull llama3.2 │
│ pulling manifest...    │
│ ████████░░░░ 67%       │
│                        │
│ [Narración overlay]    │
└────────────────────────┘
```

### Frame 3: Comparación de Modelos
```
┌────────────────────────┐
│  Llama    DeepSeek Qwen│
│  3.2:1b   R1:1.5b  2.5 │
│                        │
│  [Icono]  [Icono] [❤️] │
│  Balance  Razón   Speed│
└────────────────────────┘
```

### Frame 4: Gráfico de Resultados
```
┌────────────────────────┐
│ Tiempo de Respuesta    │
│                        │
│ Llama  ████████░░ 4.5s │
│ Deep   ██████████ 5.9s │
│ Qwen   ████░░░░░░ 1.8s │
└────────────────────────┘
```

### Frame 5: Conversación
```
┌────────────────────────┐
│ Open WebUI             │
│                        │
│ 👤: [Tu pregunta]      │
│                        │
│ 🤖: [Respuesta modelo] │
│     [...]              │
└────────────────────────┘
```

### Frame 6: Conclusión
```
┌────────────────────────┐
│ CONCLUSIONES           │
│                        │
│ ✓ Llama: Versátil      │
│ ✓ DeepSeek: Razonam.   │
│ ✓ Qwen: Eficiencia     │
│                        │
│ ¡Gracias!              │
└────────────────────────┘
```

---

## ALTERNATIVAS DE FORMATO

### Opción A: Video Estilo Tutorial
- Pantalla completa mostrando ejecución
- Narración explicativa
- Pausas para leer resultados
- **Ventaja:** Didáctico, fácil de seguir
- **Desventaja:** Puede ser lento

### Opción B: Video Estilo Presentación
- Slides con bullet points
- Insertos de código/resultados
- Ritmo más rápido
- **Ventaja:** Profesional, dinámico
- **Desventaja:** Menos técnico

### Opción C: Híbrido (Recomendado)
- Intro/conclusión con slides
- Cuerpo con screencast
- Gráficos insertados sobre video
- **Ventaja:** Balancea pedagogía y ritmo
- **Desventaja:** Más trabajo de edición

---

## RECURSOS ADICIONALES

### Música Sugerida (Royalty-Free):

**Intro/Outro:**
- "Inspired" by Kevin MacLeod (upbeat, tech)
- "Cipher" by Kevin MacLeod (tech, mysterious)

**Secciones Técnicas:**
- "Investigations" by Kevin MacLeod (low, subtle)
- "Deliberate Thought" by Kevin MacLeod (focus)

**Conclusión:**
- "Wallpaper" by Kevin MacLeod (optimista)

### Efectos de Sonido:

- Transición suave: "Whoosh" sutil
- Aparición de gráficos: "Pop" ligero
- Highlight de código: "Click" suave
- Éxito/logro: "Ding" corto

**Fuente:** Freesound.org, Zapsplat.com

---

**¡Mucha suerte con tu video! 🎥🚀**
