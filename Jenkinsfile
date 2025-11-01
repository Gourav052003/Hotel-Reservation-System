pipeline{
    agent any


    stages{
        stage('cloninig github repo to jenkins container'){
            steps{
                 script{
                    echo "cloninig github repo to jenkins container......................."
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'GitHub-Token', url: 'https://github.com/Gourav052003/Hotel-Reservation-System.git']])
                 }
            }
        }
    }
}