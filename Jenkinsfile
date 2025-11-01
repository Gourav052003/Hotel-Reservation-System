pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = "polynomial-text-474610-n0"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"

    }

    stages{


        stage('cloninig github repo to jenkins container'){
            steps{
                 script{
                    echo "cloninig github repo to jenkins container......................."
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub-Token', url: 'https://github.com/Gourav052003/Hotel-Reservation-System.git']])
                 }
            }
        }


        stage('Python environment setup'){
            steps{
                 script{
                    echo 'Python environemnt is creating .......................'
                    sh '''
                        python -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip 
                        pip install -e .
                    '''
                 }
            }
        }




        stage('Building and Pushing Docker Image to GCR'){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo "Building and Pushing Docker Image to GCR......................."
                        sh '''
                            export PATH=$PATH:${GCLOUD_PATH}
                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                            gcloud config set project ${GCP_PROJECT}
                            gcp auth configure-docker --quiet
                            docker build -t gcr.io/${GCP_PROJECT}/hotel-reservation-system:latest .
                            docker push gcr.io/${GCP_PROJECT}/hotel-reservation-system:latest
                        '''
                    }
                }
            }
        }



    }
}