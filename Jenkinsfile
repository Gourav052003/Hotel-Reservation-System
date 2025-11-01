pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'

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




        stage('loninig github repo to jenkins containe'){
            steps{
                 script{
                    echo "loninig github repo to jenkins containe......................."
                    sh '''
                        python -m venv ${VENV_DIR}
                        . ${VENV_DIR}/Scripts/activate
                        pip install --upgrade pip 
                        pip install -e .
                    '''
                 }
            }
        }




    }
}