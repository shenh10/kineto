

# npm relared env
curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
apt-get install -y nodejs
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
apt update && apt install yarn
yarn
npm install webpack
npm install --global cross-env



apt install -y default-jre
wget https://repo1.maven.org/maven2/io/swagger/codegen/v3/swagger-codegen-cli/3.0.25/swagger-codegen-cli-3.0.25.jar -O swagger-codegen-cli.jar
java -jar swagger-codegen-cli.jar generate -i ./src/api/openapi.yaml -l typescript-fetch -o ./src/api/generated/  --additional-properties modelPropertyNaming=original
rm ./src/api/generated/api_test.spec.ts
yarn prettier --end-of-line lf
yarn add url
python ./scripts/add_header.py ./src/api/generated/


yarn build:copy