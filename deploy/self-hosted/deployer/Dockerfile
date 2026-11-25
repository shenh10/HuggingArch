# The deployer: one container inside the Docker VM that pulls main and rolls
# the app stack (../docker-compose.yml) through the VM's Docker socket, so the
# machine running the VM executes nothing of the repository. Pinned to the
# docker CLI image, which ships the compose and buildx plugins.
FROM docker:29.10.0-cli
RUN apk add --no-cache git python3 bash iptables iptables-legacy
COPY --chmod=0755 autodeploy.py /usr/local/bin/autodeploy
ENTRYPOINT ["python3", "/usr/local/bin/autodeploy"]
CMD ["run"]
