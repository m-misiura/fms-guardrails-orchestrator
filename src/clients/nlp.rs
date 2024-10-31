/*
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

*/

use async_trait::async_trait;
use axum::http::{Extensions, HeaderMap};
use futures::{StreamExt, TryStreamExt};
use ginepro::LoadBalancedChannel;
use tonic::{metadata::MetadataMap, Code, Request};

use super::{create_grpc_client, errors::grpc_to_http_code, BoxStream, Client, Error};
use crate::{
    config::ServiceConfig,
    health::{HealthCheckResult, HealthStatus},
    pb::{
        caikit::runtime::nlp::{
            nlp_service_client::NlpServiceClient, ServerStreamingTextGenerationTaskRequest,
            TextGenerationTaskRequest, TokenClassificationTaskRequest, TokenizationTaskRequest,
        },
        caikit_data_model::nlp::{
            GeneratedTextResult, GeneratedTextStreamResult, TokenClassificationResults,
            TokenizationResults,
        },
        grpc::health::v1::{health_client::HealthClient, HealthCheckRequest, HealthCheckResponse},
    },
};

const DEFAULT_PORT: u16 = 8085;
const MODEL_ID_HEADER_NAME: &str = "mm-model-id";

#[cfg_attr(test, faux::create)]
#[derive(Clone)]
pub struct NlpClient {
    client: NlpServiceClient<LoadBalancedChannel>,
    health_client: HealthClient<LoadBalancedChannel>,
}

#[cfg_attr(test, faux::methods)]
impl NlpClient {
    pub async fn new(config: &ServiceConfig) -> Self {
        println!("Creating new NlpClient with config: {:?}", config);
        let client = create_grpc_client(DEFAULT_PORT, config, NlpServiceClient::new).await;
        let health_client = create_grpc_client(DEFAULT_PORT, config, HealthClient::new).await;
        println!("NlpClient created successfully");
        Self {
            client,
            health_client,
        }
    }

    pub async fn tokenization_task_predict(
        &self,
        model_id: &str,
        request: TokenizationTaskRequest,
        headers: HeaderMap,
    ) -> Result<TokenizationResults, Error> {
        println!("Starting tokenization task predict for model ID: {}", model_id);
        let mut client = self.client.clone();
        let request = request_with_model_id(request, model_id, headers);
        let response = client.tokenization_task_predict(request).await?.into_inner();
        println!("Received tokenization task response");
        Ok(response)
    }

    pub async fn token_classification_task_predict(
        &self,
        model_id: &str,
        request: TokenClassificationTaskRequest,
        headers: HeaderMap,
    ) -> Result<TokenClassificationResults, Error> {
        println!("Starting token classification task predict for model ID: {}", model_id);
        let mut client = self.client.clone();
        let request = request_with_model_id(request, model_id, headers);
        let response = client.token_classification_task_predict(request).await?.into_inner();
        println!("Received token classification task response");
        Ok(response)
    }

    pub async fn text_generation_task_predict(
        &self,
        model_id: &str,
        request: TextGenerationTaskRequest,
        headers: HeaderMap,
    ) -> Result<GeneratedTextResult, Error> {
        println!("Starting text generation task predict for model ID: {}", model_id);
        println!("Request details: {:?}", request);
        println!("Headers: {:?}", headers);
    
        let mut client = self.client.clone();
        let request = request_with_model_id(request, model_id, headers);
    
        match client.text_generation_task_predict(request).await {
            Ok(response) => {
                println!("Received text generation task response");
                Ok(response.into_inner())
            },
            Err(e) => {
                eprintln!("Error during text generation task predict: {:?}", e);
                eprintln!("Status code: {:?}", e.code());
                eprintln!("Metadata: {:?}", e.metadata());
                Err(e.into())
            }
        }
    }

    pub async fn server_streaming_text_generation_task_predict(
        &self,
        model_id: &str,
        request: ServerStreamingTextGenerationTaskRequest,
        headers: HeaderMap,
    ) -> Result<BoxStream<Result<GeneratedTextStreamResult, Error>>, Error> {
        println!("Starting server streaming text generation task predict for model ID: {}", model_id);
        let mut client = self.client.clone();
        let request = request_with_model_id(request, model_id, headers);
        let response_stream = client
            .server_streaming_text_generation_task_predict(request)
            .await?
            .into_inner()
            .map_err(Into::into)
            .boxed();
        println!("Received response stream for text generation task");
        Ok(response_stream)
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, Error> {
        println!("Performing health check");
        let request = tonic::Request::new(HealthCheckRequest { service: "".into() });
        let response = self.health_client.clone().check(request).await?.into_inner();
        println!("Health check response: {:?}", response);
        Ok(response)
    }
}

#[cfg_attr(test, faux::methods)]
#[async_trait]
impl Client for NlpClient {
    fn name(&self) -> &str {
        "nlp"
    }

    async fn health(&self) -> HealthCheckResult {
        println!("Performing health check for NLP client");
        let response = self.health_client.clone().check(HealthCheckRequest { service: "".into() }).await;
        let code = match response {
            Ok(_) => Code::Ok,
            Err(status) if matches!(status.code(), Code::InvalidArgument | Code::NotFound) => {
                Code::Ok
            }
            Err(status) => status.code(),
        };
        let status = if matches!(code, Code::Ok) {
            println!("NLP client is healthy");
            HealthStatus::Healthy
        } else {
            println!("NLP client is unhealthy");
            HealthStatus::Unhealthy
        };
        HealthCheckResult {
            status,
            code: grpc_to_http_code(code),
            reason: None,
        }
    }
}

fn request_with_model_id<T>(request: T, model_id: &str, headers: HeaderMap) -> Request<T> {
    println!("Creating request with model ID: {}", model_id);
    let metadata = MetadataMap::from_headers(headers);
    let mut request = Request::from_parts(metadata, Extensions::new(), request);
    request
        .metadata_mut()
        .insert(MODEL_ID_HEADER_NAME, model_id.parse().unwrap());
    println!("Request created with model ID: {}", model_id);
    request
}