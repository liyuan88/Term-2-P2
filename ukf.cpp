#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#define EPS 0.001

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_=false;
  // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

  // initial state vector
    x_ = VectorXd(5);

  // initial covariance matrix
    P_=MatrixXd(5,5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.7;

  // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    n_x_=5;

    n_aug_=n_x_+2; //augmented

    lambda_=3-n_aug_;
    
    x_aug_=VectorXd(n_aug_);

    P_aug_ = MatrixXd(7, 7);
    
    weights_=VectorXd(2*n_aug_+1);
    
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
}



UKF::~UKF() {}

/**
 * @param meas_package meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_){
       
        //Initialize state vector and augemented state vector
        x_.fill(0.0);
        
        
        //cout<<"x_aug_"<<x_aug_<<endl;
        
        //Initialize covariance matrix P_
        P_<<1,0,0,0,0,
            0,1,0,0,0,
            0,0,1000,0,0,
            0,0,0,1000,0,
            0,0,0,0,1;
        
        cout<<"P_:"<<P_<<endl;
        
        //Initialize augmented covariance matrix P_aug
        P_aug_.fill(0.0);
        P_aug_.topLeftCorner(n_x_,n_x_) = P_;
        P_aug_(5,5) = std_a_*std_a_;
        P_aug_(6,6) = std_yawdd_*std_yawdd_;
        
        
        //cout<<"P_aug_"<<P_aug_<<endl;
        
        //Initialize weights vector
        weights_(0) = lambda_/(lambda_ + n_aug_);
        
        for (int i=1; i<2*n_aug_+1; i++) {
            weights_(i) = 0.5/(n_aug_+lambda_);
        }
        
        //cout<<weights_<<endl;
        
        //Input the sensor data
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            
            // Radar
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            double rho_dot = meas_package.raw_measurements_(2);
            double px = rho*cos(phi);
            double py = rho*sin(phi);
            double vx = rho_dot*cos(phi);
            double vy = rho_dot*sin(phi);
            double v=sqrt(vx*vx+vy*vy);
            
            x_ << px,py,v,0,0;
            cout<<"Radar: x_: "<<x_<<endl;
            
        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            
            //LIDAR
            x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
            cout<<"Lidar: x_: "<<x_<<endl;
        }
        
        //Check if px and py are zero
        if (fabs(x_(0)) < EPS and fabs(x_(1)) < EPS) {
            x_(0) = EPS;
            x_(1) = EPS;
        }
        
        x_aug_.head(5)=x_;
        x_aug_(5) = 0;
        x_aug_(6) = 0;
        cout<<"Initialization x_aug_:"<<x_aug_<<endl;
        
        time_us_ = meas_package.timestamp_;
        
        is_initialized_ = true;
        
        return;
    }
    
    double delta_t=(meas_package.timestamp_-time_us_);
    delta_t/=1000000.0;
    time_us_=meas_package.timestamp_;
    
    //while (delta_t > 0.1)
    //{
    //    const double dt = 0.05;
    //    Prediction(dt);
    //    delta_t -= dt;
    //}
    
    Prediction(delta_t);
    
    cout<<"Finish Prediction"<<endl;
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateLidar(meas_package);
    } else {
        UpdateLidar(meas_package);
    }
}


void UKF::Prediction(double delta_t) {
    
    x_aug_.head(5)=x_;
    x_aug_(5) = 0;
    x_aug_(6) = 0;
    //cout<<"Prediction x_aug_:"<<x_aug_<<endl;

    
    //Initialize covariance matrix P_
    
    
    //cout<<"Prediction: P_"<<P_<<endl;
    
    //Initialize augmented covariance matrix P_aug
    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_(5,5)=std_a_*std_a_;
    P_aug_(6,6)=std_yawdd_*std_yawdd_;
    
    //cout<<"P_aug_"<<P_aug_<<endl;
    
    MatrixXd L = P_aug_.llt().matrixL();
    
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    //MatrixXd Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
    Xsig_aug.col(0)=x_aug_;
    
    //Generate sigma points
    for (int i=0;i<n_aug_;i++){
        Xsig_aug.col(i+1)=x_aug_+sqrt(lambda_+n_aug_)*L.col(i);
        Xsig_aug.col(i+1+n_aug_)=x_aug_-sqrt(lambda_+n_aug_)*L.col(i);
    }
    //cout << "Xsig_aug: "<<Xsig_aug<<endl;
    
    //Predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
        //extract values for better readability
        double px = Xsig_aug(0,i);
        double py = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);
        
        //Predicted values
        double px_p, py_p, v_p, yaw_p, yawd_p;
        
        //Avoid division by zero
        if (fabs(yawd) > EPS) {
            px_p = px + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = py + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = px + v*delta_t*cos(yaw);
            py_p = py + v*delta_t*sin(yaw);
        }
        
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v + nu_a * delta_t;
        yaw_p = yaw + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd + nu_yawdd * delta_t;
        
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
        
        //Predicted state mean
        x_.fill(0.0);
        for (int i = 0; i < 2 * n_aug_ + 1; i++) {
            x_ = x_ + weights_(i) * Xsig_pred_.col(i);
        }
        
        //predicted state covariance matrix
        P_.fill(0.0);
        for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
            
            // state difference
            VectorXd x_diff = Xsig_pred_.col(i) - x_;
            //angle normalization
            while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
            while (x_diff(3)< -M_PI) x_diff(3)+=2.*M_PI;
            
            P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
        }
    }
    cout << "Xsig_pred_: "<<Xsig_pred_<<endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param meas_package meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    int n_z=2;
    VectorXd z = meas_package.raw_measurements_;
    //cout<<"z (Lidar): "<<z<<endl;
    
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);
    
    //Sigma points to radar measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        
        
        // measurement model
        Zsig(0,i) = Xsig_pred_(0,i);
        Zsig(1,i) = Xsig_pred_(1,i);
    }
    //cout<<"Zsig (Lidar): "<<Zsig<<endl;
    
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    z_pred=Zsig * weights_;
    
    //cout<<"z_pred (Lidar): "<<z_pred<<endl;
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    S = S + R;
    
    //cout << "S (Lidar): "<<S<<endl;
    
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //residual
    VectorXd z_diff = z - z_pred;
    
    //cout << "z_diff (Lidar) "<<z_diff<<endl;
    
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
    
    //NIS Update
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}
    
    
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    
    int n_z=3;
    VectorXd z = meas_package.raw_measurements_;
    //cout<<"z (Radar): "<<z<<endl;
    
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);
    
    //Sigma points to radar measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;
        
        if (fabs(p_x)<EPS) {
            p_x=EPS;
        }
        if (fabs(p_y) < EPS) {
            p_y = EPS;
        }
        
        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
    //cout<<"Zsig (Radar): "<<Zsig<<endl;
    
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    z_pred=Zsig*weights_;
    
    //cout<<"z_pred (Radar): "<<z_pred<<endl;
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)< -M_PI) z_diff(1)+=2.*M_PI;
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
    
    S = S + R;
    //cout << "S (Radar): "<<S<<endl;
    
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    
    //VectorXd z = VectorXd(n_z);
    //z.fill(0.0);
    
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)< -M_PI) z_diff(1)+=2.*M_PI;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //residual
    VectorXd z_diff = z - z_pred;
    //cout << "z_diff (Radar) "<<z_diff<<endl;
    
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)< -M_PI) z_diff(1)+=2.*M_PI;
    
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
    
    //NIS Update
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    
}
