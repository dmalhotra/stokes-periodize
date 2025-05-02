template <class Real> const sctl::Vector<Real>& Periodize<Real>::GetProxySurf() {
  static const sctl::Vector<Real> proxy_surf = [](){
    sctl::Vector<Real> X;
    X.template Read<PrecompReal>("data/dn_equiv_surf.mat");
    return X;
  }();
  return proxy_surf;
}

template <class Real> void Periodize<Real>::EvalFarField(sctl::Vector<Real>& U_far, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& U_proxy) {
  const auto& Mbc0 = GetMat_UC2DE0();
  const auto& Mbc1 = GetMat_UC2DE1();
  const sctl::Long N = Mbc0.Dim(0);
  SCTL_ASSERT(U_proxy.Dim() == N);

  // Compute the equivalent density at proxy points
  auto proxy_density = (sctl::Matrix<Real>(1,N,(sctl::Iterator<Real>)U_proxy.begin(),false) * Mbc0) * Mbc1;

  // Evaluate the potential from proxy points at the targets Xt
  U_far = 0;
  static const sctl::Stokes3D_FxU stokeslet;
  stokeslet.template Eval<Real,true>(U_far, Xt, GetProxySurf(), sctl::Vector<Real>(), sctl::Vector<Real>(N,proxy_density.begin(),false));
}

template <class Real> const sctl::Matrix<Real>& Periodize<Real>::GetMat_UC2DE0() {
  static sctl::Matrix<Real> Mbc = [](){
    sctl::Matrix<Real> Mbc_ue2dc, M_dc2de0, M_uc2ue0, M_uc2ue1;
    M_uc2ue0.template Read<PrecompReal>("data/M_uc2ue0.mat");
    M_uc2ue1.template Read<PrecompReal>("data/M_uc2ue1.mat");
    Mbc_ue2dc.template Read<PrecompReal>("data/Mbc_ue2dc.mat");
    M_dc2de0.template Read<PrecompReal>("data/M_dc2de0.mat");
    return (M_uc2ue0 * (M_uc2ue1 * Mbc_ue2dc)) * M_dc2de0;
  }();
  return Mbc;
}

template <class Real> const sctl::Matrix<Real>& Periodize<Real>::GetMat_UC2DE1() {
  static sctl::Matrix<Real> Mbc = [](){
    sctl::Matrix<Real> M_dc2de1;
    M_dc2de1.template Read<PrecompReal>("data/M_dc2de1.mat");
    return M_dc2de1;
  }();
  return Mbc;
}

