import imagej.doctor


class TestDoctor(object):
    def test_checkup(self):
        output = []
        imagej.doctor.checkup(output.append)

        if output[-1].startswith("--> "):
            # There is some advice; let's skip past it.
            while output[-1].startswith("--> "):
                output.pop()
            assert output[-1] == "Questions and advice for you:"
        else:
            # No advice; all was well.
            assert output[-1] == "Great job! All looks good."
