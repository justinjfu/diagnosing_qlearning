from debugq.envs cimport env_wrapper

cdef class RandomObsWrapper(env_wrapper.TabularEnvWrapper):
    cdef _observations

cdef class LocalObsWrapper(env_wrapper.TabularEnvWrapper):
    cdef _observations

cdef class OneHotObsWrapper(env_wrapper.TabularEnvWrapper):
    cdef int dim