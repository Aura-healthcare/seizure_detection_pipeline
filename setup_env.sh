source env.sh
rm -rf conf/provisioning/datasources/datasources.yml
envsubst < "conf/template.yml" > "conf/provisioning/datasources/datasources.yml"