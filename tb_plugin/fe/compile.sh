
java -jar swagger-codegen-cli.jar generate -i ./src/api/openapi.yaml -l typescript-fetch -o ./src/api/generated/  --additional-properties modelPropertyNaming=original
rm ./src/api/generated/api_test.spec.ts
yarn prettier --end-of-line lf
yarn add url
python ./scripts/add_header.py ./src/api/generated/
yarn build:copy