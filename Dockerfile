# ------- COMMON BUILD STEPS FOR BOTH DEPLOY AND TESTS -------------------


# INSERT FROM WHERE TO WHERE


ARG HOMEDIR=/app


ENV PORT=8501 \
   HOST=0.0.0.0 \
   PYTHONPATH=${HOMEDIR} \
   STREAMLIT_SERVER_PORT=8501 \
   STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
   STREAMLIT_SERVER_HEADLESS=true \
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false


RUN apt-get update && apt-get install -y --no-install-recommends \
   curl && \
   rm -rf /var/lib/apt/lists/*


RUN useradd -d ${HOMEDIR} app


EXPOSE 8501


WORKDIR ${HOMEDIR}


COPY requirements.txt ${HOMEDIR}/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY . ${HOMEDIR}


# fix permissions
RUN find ${HOMEDIR} -exec chown app {} \; \
   && find ${HOMEDIR} -type d -exec chmod 755 {} \; \
   && find ${HOMEDIR} -type f -exec chmod 644 {} \; \
   && chmod 755 start.sh


USER app


HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
   CMD curl -f http://localhost:8501/_stcore/health || exit 1


CMD [ "/app/start.sh" ]


# ------- STAGE FOR LOCAL DEVELOPMENT -------------------
FROM base AS dev-image
# dev uses same deps, streamlit has built-in reload


# ------- STAGE FOR DEPLOY -------------------
FROM base AS final-image


# ------- STAGE FOR PREPPING TESTS -------------------
FROM base AS prepare-tests
RUN pip install --no-cache-dir pytest