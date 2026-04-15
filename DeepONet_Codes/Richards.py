import sys
import jax.numpy as np



def soil_parameters(soil_type):

        nvg = None
        mvg = None
        ksvg = None
        alphavg = None
        hcap = None
        thetaRvg = None
        thetaSvg = None


        if soil_type == "silt loam":
            nvg = 1.41
            mvg = 1 - 1 / nvg
            ksvg = 0.108
            alphavg = 2.0
            hcap = 1/ alphavg
            thetaRvg = 0.067
            thetaSvg = 0.45

        elif soil_type == "loam":
            nvg = 1.56
            mvg = 1 - 1 / nvg
            ksvg = 0.2496
            alphavg = 3.6
            hcap = 1/ alphavg
            thetaRvg = 0.078
            thetaSvg = 0.43
        elif soil_type == "sandy loam":
            nvg = 1.89
            mvg = 1 - 1 / nvg
            ksvg = 1.061
            alphavg = 7.5
            hcap = 1/ alphavg
            hcap = 1 / alphavg
            thetaRvg = 0.065
            thetaSvg = 0.41
        elif soil_type == "sand":
            nvg = 2.68
            mvg = 1-1/nvg
            ksvg = 7.128
            alphavg = 14.5
            hcap = 1/ alphavg
            hcap = 1/alphavg
            thetaRvg = 0.045
            thetaSvg = 0.43
        else:
            print("Invalid soil type entered.")
            sys.exit(1)

        if nvg is not None:
            print(f"Parameters for {soil_type}:")
            print(f"nvg = {nvg}")
            print(f"mvg = {mvg}")
            print(f"ksvg = {ksvg}")
            print(f"alphavg = {alphavg}")
            print(f"hcap = {hcap}")
            print(f"thetaRvg = {thetaRvg}")
            print(f"thetaSvg = {thetaSvg}")


        return nvg, mvg, ksvg, alphavg, hcap, thetaRvg, thetaSvg

def VG_model(nvg, mvg, ksvg, alphavg, hcap, thetaRvg, thetaSvg): 
    # WRC: Water retention curve
    def theta_function(h):
        term2 = 1 + np.power(np.abs(alphavg * h), nvg)
        term3 = np.power(term2, -mvg)
        result = thetaRvg + (thetaSvg - thetaRvg) * term3
        result = np.where(h > 0, thetaSvg, result)
        return result

    # HCF: Hydraulic conductivity function
    def K_function(h):
        theta_h = theta_function(h)
        term1 = np.power((theta_h - thetaRvg) / (thetaSvg - thetaRvg), 0.5)
        term2 = 1 - np.power(1 - np.power((theta_h - thetaRvg) / (thetaSvg - thetaRvg), 1/mvg), mvg)
        result = ksvg * term1 * np.power(term2, 2)
        result = np.where(h > 0, ksvg, result)
        return result

    # J-Leverett function: Psi = hcap*J(Saturation)
    def h_function(theta):
        S = (theta - thetaRvg) / (thetaSvg - thetaRvg)
        J_S = -(np.power(S, -1/mvg) - 1)**(1/nvg)
        return hcap*J_S
    
    return theta_function, K_function, h_function

