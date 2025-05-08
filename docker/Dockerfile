FROM student_mechanism_design_base
WORKDIR /app
COPY . /app
RUN gcc --version
RUN dos2unix /app/run_main.sh
RUN chmod a+x /app/run_main.sh
ENTRYPOINT ["/app/run_main.sh"]