/* A test FMU for sim_vehicle_ecu's fmu component (scensim roadmap 091 W4): an FMI 2.0
 * co-simulation unit with a VESC-like speed loop and servo, built by the tests (test_fmu.py).
 * Not a car model: it only gives each mapping, the reset, the seed and the step a value to check.
 *
 *   err    = speed_ref - wheel_radius * (wheel_rl + wheel_rr) / 2
 *   integ += err * h
 *   torque = kp * err + ki * integ + noise_std * u(seed)   (u uniform in [-1, 1), an LCG)
 *   torque_rl = torque_rr = torque / 2;  brake = brake_on ? 2 : 0
 *   steer  = (servo - steer_offset) * steer_gain;  erpm = gear * (wheel_rl + wheel_rr) / 2
 */
#include <stdlib.h>
#include <string.h>
#include "fmi2Functions.h"

enum { SPEED_REF, SERVO, WHEEL_RL, WHEEL_RR, TORQUE_RL, TORQUE_RR, STEER, ERPM, KP, KI,
       WHEEL_RADIUS, GEAR, NOISE_STD, STEER_GAIN, STEER_OFFSET, BRAKE, N_REAL };
enum { STEPS, SEED, N_INT };
enum { BRAKE_ON, N_BOOL };

typedef struct {
    fmi2Real r[N_REAL];
    fmi2Integer i[N_INT];
    fmi2Boolean b[N_BOOL];
    fmi2Real integ, time;
    unsigned long long lcg;
    fmi2CallbackFunctions cb;
    fmi2String name;
} Racer;

static void defaults(Racer *m) {
    memset(m->r, 0, sizeof m->r);
    memset(m->i, 0, sizeof m->i);
    memset(m->b, 0, sizeof m->b);
    m->r[KP] = 0.5; m->r[KI] = 0.1; m->r[WHEEL_RADIUS] = 0.05; m->r[GEAR] = 20.0;
    m->r[STEER_GAIN] = -1.2; m->r[STEER_OFFSET] = 0.53;
    m->integ = 0.0; m->time = 0.0;
}

static double uniform(Racer *m) {
    m->lcg = m->lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((double)(m->lcg >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

static void outputs(Racer *m) {
    m->r[STEER] = (m->r[SERVO] - m->r[STEER_OFFSET]) * m->r[STEER_GAIN];
    m->r[ERPM] = m->r[GEAR] * (m->r[WHEEL_RL] + m->r[WHEEL_RR]) / 2.0;
    m->r[BRAKE] = m->b[BRAKE_ON] ? 2.0 : 0.0;
}

const char *fmi2GetTypesPlatform(void) { return fmi2TypesPlatform; }
const char *fmi2GetVersion(void) { return fmi2Version; }
fmi2Status fmi2SetDebugLogging(fmi2Component c, fmi2Boolean on, size_t n, const fmi2String cat[]) {
    return fmi2OK;
}

fmi2Component fmi2Instantiate(fmi2String name, fmi2Type type, fmi2String guid, fmi2String res,
                              const fmi2CallbackFunctions *cb, fmi2Boolean visible,
                              fmi2Boolean logging) {
    if (type != fmi2CoSimulation || strcmp(guid, "{8c4e810f-3df3-4a00-8276-176fa3c9f000}"))
        return NULL;
    Racer *m = calloc(1, sizeof(Racer));
    m->cb = *cb;
    m->name = name;
    defaults(m);
    return m;
}
void fmi2FreeInstance(fmi2Component c) { free(c); }
fmi2Status fmi2SetupExperiment(fmi2Component c, fmi2Boolean tolDef, fmi2Real tol, fmi2Real t0,
                               fmi2Boolean stopDef, fmi2Real stop) {
    ((Racer *)c)->time = t0;
    return fmi2OK;
}
fmi2Status fmi2EnterInitializationMode(fmi2Component c) { return fmi2OK; }
fmi2Status fmi2ExitInitializationMode(fmi2Component c) {
    Racer *m = c;
    m->lcg = (unsigned long long)m->i[SEED] * 2654435761ULL + 1ULL;
    outputs(m);
    return fmi2OK;
}
fmi2Status fmi2Terminate(fmi2Component c) { return fmi2OK; }
fmi2Status fmi2Reset(fmi2Component c) { defaults(c); return fmi2OK; }

#define GETSET(T, FIELD, N, NAME)                                                           \
    fmi2Status fmi2Get##NAME(fmi2Component c, const fmi2ValueReference vr[], size_t n,     \
                             T v[]) {                                                      \
        for (size_t k = 0; k < n; k++) {                                                    \
            if (vr[k] >= N) return fmi2Error;                                               \
            v[k] = ((Racer *)c)->FIELD[vr[k]];                                              \
        }                                                                                   \
        return fmi2OK;                                                                      \
    }                                                                                       \
    fmi2Status fmi2Set##NAME(fmi2Component c, const fmi2ValueReference vr[], size_t n,     \
                             const T v[]) {                                                \
        for (size_t k = 0; k < n; k++) {                                                    \
            if (vr[k] >= N) return fmi2Error;                                               \
            ((Racer *)c)->FIELD[vr[k]] = v[k];                                              \
        }                                                                                   \
        return fmi2OK;                                                                      \
    }
GETSET(fmi2Real, r, N_REAL, Real)
GETSET(fmi2Integer, i, N_INT, Integer)
GETSET(fmi2Boolean, b, N_BOOL, Boolean)

fmi2Status fmi2GetString(fmi2Component c, const fmi2ValueReference vr[], size_t n, fmi2String v[]) {
    return fmi2Error;
}
fmi2Status fmi2SetString(fmi2Component c, const fmi2ValueReference vr[], size_t n,
                         const fmi2String v[]) {
    return fmi2Error;
}
fmi2Status fmi2GetFMUstate(fmi2Component c, fmi2FMUstate *s) { return fmi2Error; }
fmi2Status fmi2SetFMUstate(fmi2Component c, fmi2FMUstate s) { return fmi2Error; }
fmi2Status fmi2FreeFMUstate(fmi2Component c, fmi2FMUstate *s) { return fmi2Error; }
fmi2Status fmi2SerializedFMUstateSize(fmi2Component c, fmi2FMUstate s, size_t *n) { return fmi2Error; }
fmi2Status fmi2SerializeFMUstate(fmi2Component c, fmi2FMUstate s, fmi2Byte b[], size_t n) {
    return fmi2Error;
}
fmi2Status fmi2DeSerializeFMUstate(fmi2Component c, const fmi2Byte b[], size_t n, fmi2FMUstate *s) {
    return fmi2Error;
}
fmi2Status fmi2GetDirectionalDerivative(fmi2Component c, const fmi2ValueReference u[], size_t nu,
                                        const fmi2ValueReference z[], size_t nz,
                                        const fmi2Real dv[], fmi2Real dz[]) {
    return fmi2Error;
}
fmi2Status fmi2SetRealInputDerivatives(fmi2Component c, const fmi2ValueReference vr[], size_t n,
                                       const fmi2Integer o[], const fmi2Real v[]) {
    return fmi2Error;
}
fmi2Status fmi2GetRealOutputDerivatives(fmi2Component c, const fmi2ValueReference vr[], size_t n,
                                        const fmi2Integer o[], fmi2Real v[]) {
    return fmi2Error;
}

fmi2Status fmi2DoStep(fmi2Component c, fmi2Real t, fmi2Real h, fmi2Boolean noPrior) {
    Racer *m = c;
    if (!(h > 0.0)) {
        m->cb.logger(m->cb.componentEnvironment, m->name, fmi2Error, "logError",
                     "step size %g is not positive", h);
        return fmi2Error;
    }
    double err = m->r[SPEED_REF] - m->r[WHEEL_RADIUS] * (m->r[WHEEL_RL] + m->r[WHEEL_RR]) / 2.0;
    m->integ += err * h;
    double torque = m->r[KP] * err + m->r[KI] * m->integ + m->r[NOISE_STD] * uniform(m);
    m->r[TORQUE_RL] = m->r[TORQUE_RR] = torque / 2.0;
    outputs(m);
    m->i[STEPS] += 1;
    m->time = t + h;
    return fmi2OK;
}
fmi2Status fmi2CancelStep(fmi2Component c) { return fmi2Error; }
fmi2Status fmi2GetStatus(fmi2Component c, const fmi2StatusKind s, fmi2Status *v) { return fmi2Error; }
fmi2Status fmi2GetRealStatus(fmi2Component c, const fmi2StatusKind s, fmi2Real *v) { return fmi2Error; }
fmi2Status fmi2GetIntegerStatus(fmi2Component c, const fmi2StatusKind s, fmi2Integer *v) {
    return fmi2Error;
}
fmi2Status fmi2GetBooleanStatus(fmi2Component c, const fmi2StatusKind s, fmi2Boolean *v) {
    return fmi2Error;
}
fmi2Status fmi2GetStringStatus(fmi2Component c, const fmi2StatusKind s, fmi2String *v) {
    return fmi2Error;
}
