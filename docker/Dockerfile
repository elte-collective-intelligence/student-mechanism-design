FROM mechanism_design_base
WORKDIR /app
COPY . /app
RUN gcc --version
RUN chmod a+x /app/run_main.sh
ENTRYPOINT ["/app/run_main.sh"]